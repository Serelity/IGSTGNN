"""One open stock--flow step whose returned outflow is the prediction.

This kernel uses a common model flow unit per *nominal* time unit.  ``stock``
and ``storage_limit`` use that flow unit times nominal time.  Neither these
units nor a station-to-compartment mapping are certified as real vehicles.
The kernel proves a conditional model ledger, not a calibrated CTM or CFL
claim.  All sending and receiving bounds use the same old stock.  There is no
post-update clipping or unconstrained output readout.
"""
import math
from numbers import Real

import torch
from torch.nn import functional as F


SCHEMA = 'open_stock_flow_nominal_step_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _maximum(values):
    return values.max() if values.numel() else values.new_zeros(())


def _scale(limit, total):
    """A shared proportional allocation; empty demand keeps scale one."""
    positive = total > 0
    # Reserve a few ulps only at a saturated allocation.  This is an explicit
    # pre-flow numerical margin, not a post-stock clamp: aggregated rounded
    # fluxes must not send more than old stock or fill past old free space.
    saturated = positive & (total >= limit)
    finfo = torch.finfo(total.dtype)
    safe_limit = (limit * (1-8*finfo.eps) - 8*finfo.tiny).clamp_min(0)
    # Never evaluate an unused limit/tiny-demand ratio: its forward may be
    # infinity and its zero-weight backward may become 0*infinity -> NaN.
    # At an active division the existing absolute allocation reserve makes
    # safe_limit exactly zero for subnormal bounds.  A normal-sized denominator
    # also keeps reciprocal gradients representable.  Such negligible bounded
    # requests are conservatively withheld, never sent above their limit.
    denominator = torch.where(saturated, total.clamp_min(finfo.tiny), torch.ones_like(total))
    numerator = torch.where(saturated, safe_limit, torch.ones_like(total))
    return torch.minimum(torch.ones_like(total), numerator / denominator)


def _aggregate(edge_values, incidence):
    # Dense incidence matmul avoids nondeterministic CUDA scatter atomics.
    return (edge_values.squeeze(-1) @ incidence).unsqueeze(-1)


def step(stock, storage_limit, node_service_rate, edge_demand_rate,
         external_backlog, edge_index, dt):
    """Advance one simultaneous, open-system stock--flow step.

    Node inputs are floating tensors [batch, nodes, 1].  Demand and backlog
    are [batch, edges, 1]; edge_index is torch.long [2, edges].  An endpoint
    of -1 denotes the exterior.  Exterior-to-exterior edges are forbidden.
    Backlog is allowed only on entry edges (-1 -> node).  On an entry edge,
    demand is the NEW arrival rate; on other edges it is a requested rate.

    Source allocations share min(old_stock / dt, service_rate); destination
    allocations share (storage_limit - old_stock) / dt.  Unaccepted entry
    demand remains in external_backlog_next, including old entry backlog.
    All returned rates and stocks retain autograd.  Only ``stats`` detach.
    """
    _require(isinstance(stock, torch.Tensor), 'stock must be a tensor')
    _require(stock.ndim == 3 and stock.shape[-1] == 1 and min(stock.shape[:2]) > 0,
             'stock axes must be [positive batch, positive nodes, 1]')
    _require(stock.dtype in (torch.float32, torch.float64),
             'stock-flow requires float32 or float64')
    _require(isinstance(dt, Real) and not isinstance(dt, bool)
             and math.isfinite(float(dt)) and float(dt) > 0, 'dt must be finite and positive')
    dt = float(dt)
    _require(isinstance(edge_index, torch.Tensor) and edge_index.dtype == torch.long
             and edge_index.ndim == 2 and edge_index.shape[0] == 2,
             'edge_index must be Long[2, edges]')
    _require(edge_index.device == stock.device, 'edge_index device mismatch')
    batch, nodes, _ = stock.shape
    edges = edge_index.shape[1]
    for name, value, shape in (
        ('storage_limit', storage_limit, stock.shape),
        ('node_service_rate', node_service_rate, stock.shape),
        ('edge_demand_rate', edge_demand_rate, (batch, edges, 1)),
        ('external_backlog', external_backlog, (batch, edges, 1)),
    ):
        _require(isinstance(value, torch.Tensor) and value.shape == shape,
                 name + ' axes mismatch')
        _require(value.dtype == stock.dtype and value.device == stock.device,
                 name + ' dtype/device mismatch')
    for name, value in (('stock', stock), ('storage_limit', storage_limit),
                        ('node_service_rate', node_service_rate),
                        ('edge_demand_rate', edge_demand_rate),
                        ('external_backlog', external_backlog)):
        _require(bool(torch.isfinite(value).all()) and bool((value >= 0).all()),
                 name + ' must be finite and nonnegative')
    _require(bool((stock <= storage_limit).all()), 'stock exceeds storage_limit')
    _require(bool(((edge_index >= -1) & (edge_index < nodes)).all()),
             'edge endpoint is outside [-1, nodes)')
    src, dst = edge_index
    entry, exit_edge = src == -1, dst == -1
    _require(not bool((entry & exit_edge).any()), 'exterior-to-exterior edge is forbidden')
    _require(bool((external_backlog[:, ~entry] == 0).all()),
             'external_backlog must be zero on non-entry edges')

    outgoing_incidence = F.one_hot(src.clamp_min(0), nodes).to(stock.dtype)
    outgoing_incidence = outgoing_incidence * (~entry)[:, None].to(stock.dtype)
    incoming_incidence = F.one_hot(dst.clamp_min(0), nodes).to(stock.dtype)
    incoming_incidence = incoming_incidence * (~exit_edge)[:, None].to(stock.dtype)

    stock_rate = stock / dt
    sending_limit = torch.minimum(stock_rate, node_service_rate)
    receiving_limit = (storage_limit - stock) / dt
    requested = torch.where(entry[None, :, None],
                            edge_demand_rate + external_backlog / dt, edge_demand_rate)
    requested_outflow = _aggregate(requested, outgoing_incidence)
    source_factor = _scale(sending_limit, requested_outflow)
    source_factor_edges = torch.where(entry[None, :, None], 1.,
                                      source_factor[:, src.clamp_min(0)])
    source_allocated = requested * source_factor_edges
    offered_inflow = _aggregate(source_allocated, incoming_incidence)
    _require(bool(torch.isfinite(stock_rate).all() & torch.isfinite(requested).all()
                  & torch.isfinite(requested_outflow).all() & torch.isfinite(offered_inflow).all()),
             'stock-flow intermediate rate overflowed')
    destination_factor = _scale(receiving_limit, offered_inflow)
    destination_factor_edges = torch.where(exit_edge[None, :, None], 1.,
                                           destination_factor[:, dst.clamp_min(0)])
    flux = source_allocated * destination_factor_edges
    inflow = _aggregate(flux, incoming_incidence)
    outflow = _aggregate(flux, outgoing_incidence)
    stock_next = stock + dt * (inflow - outflow)
    # Entry source scale is exactly one.  The remaining-fraction form is
    # algebraically backlog + dt*(new arrivals - entry flux), and preserves
    # an EXACT zero when every arrival is accepted without subtraction loss.
    external_backlog_next = torch.where(entry[None, :, None],
        (external_backlog + dt * edge_demand_rate) * (1-destination_factor_edges),
        torch.zeros_like(external_backlog))

    # Validate the RAW result, retaining normal roundoff in the returned tensor.
    # No clamp may erase a failed ledger or an invalid time discretization.
    eps = torch.finfo(stock.dtype).eps
    magnitude = torch.maximum(storage_limit.max(), stock.new_tensor(1.))
    tolerance = 64 * eps * magnitude
    _require(bool(torch.isfinite(stock_next).all() & torch.isfinite(flux).all()
                  & torch.isfinite(external_backlog_next).all()), 'stock-flow output is non-finite')
    _require(bool((stock_next >= -tolerance).all()
                  & (stock_next <= storage_limit + tolerance).all()),
             'unclipped stock-flow result escaped storage bounds')
    backlog_scale = torch.maximum(_maximum(external_backlog + dt * edge_demand_rate),
                                  stock.new_tensor(1.))
    _require(bool((external_backlog_next >= -64 * eps * backlog_scale).all()),
             'unclipped external backlog became negative')

    with torch.no_grad():
        stock64, next64 = stock.double(), stock_next.double()
        in64, out64, flux64 = inflow.double(), outflow.double(), flux.double()
        node_balance = next64 - stock64 - dt * (in64 - out64)
        boundary_rate = flux64[:, entry].sum(1) - flux64[:, exit_edge].sum(1)
        global_balance = (next64 - stock64).sum(1) - dt * boundary_rate
        external_balance = (external_backlog_next.double() - external_backlog.double()
                            - dt * (edge_demand_rate.double() - flux64))[:, entry]
        source_busy = requested_outflow > sending_limit
        service_bound = source_busy & (node_service_rate <= stock_rate)
        stock_bound = source_busy & (stock_rate < node_service_rate)
        receiving_bound = offered_inflow > receiving_limit
        # A pre-receiving source restriction need not still bind final flow.
        service_active = service_bound & ((outflow - node_service_rate).abs()
            <= 16 * eps * torch.maximum(node_service_rate, torch.ones_like(node_service_rate)))
        stats = dict(
            node_positions=stock.new_tensor(stock.numel()).detach(),
            edge_positions=stock.new_tensor(flux.numel()).detach(),
            demand_limited_count=(~source_busy).sum().detach(),
            stock_limited_count=stock_bound.sum().detach(),
            service_limited_count=service_bound.sum().detach(),
            receiving_limited_count=receiving_bound.sum().detach(),
            service_active_count=service_active.sum().detach(),
            node_balance_abs_max=_maximum(node_balance.abs()).detach(),
            global_balance_abs_max=_maximum(global_balance.abs()).detach(),
            external_balance_abs_max=_maximum(external_balance.abs()).detach(),
            sending_violation_max=_maximum((outflow-sending_limit).clamp_min(0)).detach(),
            receiving_violation_max=_maximum((inflow-receiving_limit).clamp_min(0)).detach(),
            capacity_violation_max=_maximum((outflow-node_service_rate).clamp_min(0)).detach(),
            stock_lower_violation_max=_maximum((-stock_next).clamp_min(0)).detach(),
            stock_upper_violation_max=_maximum((stock_next-storage_limit).clamp_min(0)).detach(),
            external_backlog_lower_violation_max=_maximum((-external_backlog_next).clamp_min(0)).detach(),
            accepted_entry_rate_sum=flux[:, entry].sum().detach(),
            exit_rate_sum=flux[:, exit_edge].sum().detach(),
            external_backlog_sum=external_backlog_next.sum().detach(),
            allocation_relative_margin=stock.new_tensor(8*eps).detach(),
            allocation_absolute_rate_margin=stock.new_tensor(8*torch.finfo(stock.dtype).tiny).detach(),
        )
    return dict(schema=SCHEMA, state=stock_next, stock=stock_next, stock_next=stock_next,
                flux=flux, flux_rate=flux, inflow=inflow, outflow=outflow,
                node_inflow_rate=inflow, node_outflow_rate=outflow,
                external_backlog=external_backlog_next,
                external_backlog_next=external_backlog_next,
                requested_edge_rate=requested, source_allocated_edge_rate=source_allocated,
                sending_limit=sending_limit, receiving_limit=receiving_limit,
                source_factor=source_factor, destination_factor=destination_factor,
                stats=stats, diagnostics=stats)
