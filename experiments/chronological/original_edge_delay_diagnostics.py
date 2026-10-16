"""Read-only lag/support telemetry and paired descriptive analysis.

No optimizer, training loader, target-conditioned routing or model mutation.
Support means nonnegative road-path coefficient mass, not physical capacity.
"""
from contextlib import contextmanager
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments.chronological.original_edge_delay import HistoricalDelayDif, lag_kernels, road_messages
from experiments.chronological.original_igstgnn_ablation_metrics import _case_summary, _window_scores


@contextmanager
def existing_run_read_lock(directory):
    """Share the existing launcher lock without creating or writing a file."""
    path = Path(directory)/'active.lock'
    with path.open('rb') as stream:
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBRLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('Run is being written by another launcher: '+str(directory)) from error
        yield


def independent_output(path, roots):
    path = Path(path).resolve()
    for root in roots:
        root = Path(root).resolve()
        if path == root or root in path.parents or path in root.parents:
            raise ValueError('Diagnostic output must be separate from input runs')
    if path.exists():
        raise ValueError('Diagnostic output already exists; use a fresh directory')
    return path


class LagTelemetry:
    """Hooks retain references only; all reductions happen after timed forward."""
    def __init__(self, model, bins=2048):
        self.bins, self.pending, self.lags, self.support = bins, [], {}, {}
        self.handles = []
        for layer_index, layer in enumerate(model.layers):
            wrapper = layer.dif_layer
            if isinstance(wrapper, HistoricalDelayDif):
                def capture(module, args, kwargs, output, number=layer_index):
                    if module.mode != 'NO_ALIGN':
                        history = kwargs['gated_history_data'] if 'gated_history_data' in kwargs else args[1]
                        for graph, estimator in enumerate(module.delays):
                            self.pending.append((number, graph, module.original.localized_st_conv.k_t,
                                history.shape[1], estimator, estimator.last_delay))
                self.handles.append(wrapper.register_forward_hook(capture, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def consume(self, input_length):
        pending, self.pending = self.pending, []
        for layer, graph, window, history, estimator, delay in pending:
            values = delay.detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(values).all() or (values < 0).any() or (values > estimator.max_delay).any():
                raise ValueError('Invalid observed lag')
            batch, edges = values.shape
            key = (layer, graph)
            item = self.lags.setdefault(key, dict(layer=layer, graph=graph, mode=estimator.mode,
                samples=0, positions=0, minimum=None, maximum=None, sum=0., squared_sum=0.,
                edge_mean=np.zeros(edges), edge_m2=np.zeros(edges), edge_std_sum=0.,
                histogram=np.zeros(self.bins, np.int64), max_delay=estimator.max_delay,
                shuffle_changed_positions=0, shuffle_absolute_change_sum=0.))
            old_samples = item['samples']
            item['samples'] += batch
            item['positions'] += values.size
            if values.size:
                item['minimum'] = float(values.min()) if item['minimum'] is None else min(item['minimum'], float(values.min()))
                item['maximum'] = float(values.max()) if item['maximum'] is None else max(item['maximum'], float(values.max()))
                item['sum'] += float(values.sum())
                item['squared_sum'] += float(np.square(values).sum())
                chunk_mean = values.mean(0)
                difference = chunk_mean-item['edge_mean']
                item['edge_m2'] += np.square(values-chunk_mean).sum(0)+np.square(difference)*old_samples*batch/item['samples']
                item['edge_mean'] += difference*batch/item['samples']
                item['edge_std_sum'] += float(values.std(1).sum())
                item['histogram'] += np.histogram(values, bins=self.bins, range=(0, estimator.max_delay))[0]
                if estimator.mode == 'SHUFFLE':
                    change = np.abs(values-np.roll(values, -1, axis=-1))
                    item['shuffle_changed_positions'] += int((change > 0).sum())
                    item['shuffle_absolute_change_sum'] += float(change.sum())
            self._support(layer, graph, window, history, input_length, estimator, delay)

    @torch.no_grad()
    def _support(self, layer, graph, window, history, input_length, estimator, delay):
        length = history-window+1
        if length < 1 or input_length < history:
            raise ValueError('Invalid historical time axis')
        # Evaluate the mathematical interpolation mass in float64 using the
        # actually observed FP32 lag values; this is not hidden-feature mass.
        adjacency = estimator.adjacency.to(torch.float64)
        dense = adjacency.new_zeros((len(delay), *adjacency.shape))
        dense[:, estimator.receiver, estimator.source] = delay.to(torch.float64)
        kernels = lag_kernels(adjacency, dense, estimator.max_delay)
        ones = adjacency.new_ones((len(delay), length, len(adjacency), 1))
        available = road_messages(ones, kernels, orders=2)
        references = [adjacency.sum(-1)-adjacency.diagonal(),
            adjacency @ adjacency.sum(-1)-(adjacency*adjacency.T).sum(-1)]
        for order, (observed, reference) in enumerate(zip(available, references), 1):
            ref = reference.detach().cpu().numpy()
            val = observed.squeeze(-1).detach().cpu().numpy()
            tolerance = 1e-10*(1+float(np.max(np.abs(ref), initial=0)))
            if (ref < -tolerance).any() or (val < -tolerance).any() or (val-ref[None, None, :] > tolerance).any():
                raise ValueError('Available road-path mass outside its complete-history reference')
            ref, val = np.maximum(ref, 0), np.maximum(val, 0)
            valid = ref > tolerance
            for t in range(length):
                key = (layer, graph, order, t)
                item = self.support.setdefault(key, dict(layer=layer, graph=graph, order=order,
                    historical_input_length=history, summary_length=length, summary_index=t,
                    window_end_input_index=input_length-length+t, samples=0, receiver_positions=0,
                    nonzero_reference_positions=0, zero_reference_positions=0,
                    reference_mass_sum=0., available_mass_sum=0., fraction_sum=0.,
                    fraction_minimum=None, fraction_maximum=None, positions_with_loss=0))
                ratio = np.clip(val[:, t, valid]/ref[valid], 0, 1)
                item['samples'] += len(delay)
                item['receiver_positions'] += len(delay)*len(ref)
                item['nonzero_reference_positions'] += ratio.size
                item['zero_reference_positions'] += len(delay)*int((~valid).sum())
                item['reference_mass_sum'] += float(ref.sum())*len(delay)
                item['available_mass_sum'] += float(val[:, t].sum())
                item['fraction_sum'] += float(ratio.sum())
                item['positions_with_loss'] += int((ratio < 1-1e-10).sum())
                if ratio.size:
                    lo, hi = float(ratio.min()), float(ratio.max())
                    item['fraction_minimum'] = lo if item['fraction_minimum'] is None else min(lo, item['fraction_minimum'])
                    item['fraction_maximum'] = hi if item['fraction_maximum'] is None else max(hi, item['fraction_maximum'])

    def finish(self):
        lag_rows, support_rows = [], []
        for item in self.lags.values():
            row = {k: v for k, v in item.items() if k not in ('edge_mean', 'edge_m2', 'histogram')}
            count, samples = row['positions'], row['samples']
            row['mean'] = row['sum']/count if count else None
            row['edge_std_per_sample_mean'] = row['edge_std_sum']/samples if count else None
            variance = item['edge_m2']/samples
            row['sample_std_per_edge_mean'] = float(np.sqrt(np.maximum(variance, 0)).mean()) if count else None
            row['quantile_intervals'] = {}
            if count:
                cdf = np.cumsum(item['histogram'])
                for q in (.05, .5, .95):
                    i = min(self.bins-1, int(np.searchsorted(cdf, math.ceil(q*count))))
                    row['quantile_intervals'][str(q)] = [i*row['max_delay']/self.bins, (i+1)*row['max_delay']/self.bins]
            row['histogram_counts'] = item['histogram'].tolist()
            row['histogram_bins'] = self.bins
            lag_rows.append(row)
        for item in self.support.values():
            row = dict(item)
            n, ref = row['nonzero_reference_positions'], row['reference_mass_sum']
            row['available_fraction_receiver_mean'] = row['fraction_sum']/n if n else None
            row['missing_weight_fraction'] = max(0., 1-row['available_mass_sum']/ref) if ref else None
            support_rows.append(row)
        return dict(lags=lag_rows, boundary_support=support_rows,
            quantile_method='2048 fixed-width bins; reported enclosing intervals, not exact quantiles',
            support_scope='float64 coefficient mass from observed lags; final diagonal removed at each spatial order',
            physical_capacity_identified=False)


def summarize_cases(cases, target, indices, stations, roads, hard_ids, pairs, response_tol=.001):
    """Reference error minus candidate error; fixed original SOURCE hard IDs."""
    target = np.asarray(target)
    null = float(target.min()) if target.min() < 1 else 0.
    mask = target != null
    if target.ndim != 3 or not np.isfinite(target).all() or not mask.any() or (target[mask] <= 0).any():
        raise ValueError('Expected original finite positive-valid-target mask')
    if (np.asarray(indices).shape != (len(target),) or len(set(indices)) != len(indices)
            or np.asarray(stations).shape != (target.shape[2],) or len(set(stations)) != len(stations)):
        raise ValueError('Invalid sample/station axis')
    if len(set(hard_ids)) != len(hard_ids) or not set(hard_ids).issubset(set(indices)):
        raise ValueError('Fixed hard IDs differ from the original sample axis')
    groups = dict(all_nodes=mask, fixed_hard=mask & np.isin(indices, hard_ids)[:, None, None])
    for lo, hi in zip((0, 5, 20, 50, 100, 200), (5, 20, 50, 100, 200, float('inf'))):
        groups[f'flow:[{lo},{hi})'] = mask & (target >= lo) & (target < hi)
    for name, members in roads.items():
        groups['road:'+name] = mask & np.isin(np.arange(target.shape[2]), members)[None, None, :]
    bias_case = 'DYNAMIC_LAST_NO_ALIGN'
    if bias_case in cases:
        groups['bias:over'] = mask & (cases[bias_case] > target)
        groups['bias:under'] = mask & (cases[bias_case] < target)
        groups['bias:exact'] = mask & (cases[bias_case] == target)
    metrics, comparisons = {}, {}
    def summary(prediction, selected):
        error = prediction.astype(np.float64)-target.astype(np.float64)
        counts = selected.sum((0, 2))
        return _case_summary(dict(absolute=np.where(selected, np.abs(error), 0).sum((0, 2)),
            squared=np.where(selected, error*error, 0).sum((0, 2)),
            percentage=np.divide(np.abs(error), target, out=np.zeros_like(error), where=selected).sum((0, 2))), counts)
    for name, prediction in cases.items():
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError('Unpaired or nonfinite prediction: '+name)
        metrics[name] = summary(prediction, mask)
        scores = _window_scores(prediction, target, mask)
        complete = np.isfinite(scores)
        selected = np.isin(indices, hard_ids) & complete
        metrics[name].update(window_p95=float(np.quantile(scores[complete], .95)) if complete.any() else None,
            fixed_hard_window_mae=float(scores[selected].mean()) if selected.any() else None)
    for reference, candidate in pairs:
        error_r = np.abs(cases[reference].astype(np.float64)-target)
        error_c = np.abs(cases[candidate].astype(np.float64)-target)
        gain = error_r-error_c
        change = cases[candidate].astype(np.float64)-cases[reference].astype(np.float64)
        group_rows = {}
        for name, selected in groups.items():
            n = int(selected.sum())
            counts = selected.sum((0, 2))
            per_horizon = np.where(selected, gain, 0).sum((0, 2))
            pos, neg = float(np.maximum(gain[selected], 0).sum()), float(np.maximum(-gain[selected], 0).sum())
            row = dict(count=n, mae_gain_pooled=float(gain[selected].mean()) if n else None,
                mae_gain_macro=float((per_horizon/counts).mean()) if (counts > 0).all() else None,
                global_mae_contribution=float(per_horizon.sum()/mask.sum()),
                positive_gain=pos/n if n else None, negative_gain=neg/n if n else None,
                cancellation_fraction=1-abs(pos-neg)/(pos+neg) if pos+neg else (0. if n else None),
                prediction_signed_mean=float(change[selected].mean()) if n else None,
                prediction_absolute_mean=float(np.abs(change[selected]).mean()) if n else None,
                prediction_response_fraction=float((np.abs(change[selected]) > response_tol).mean()) if n else None)
            row['horizons'] = [dict(step=i+1, count=int(counts[i]),
                mae_gain=float(per_horizon[i]/counts[i]) if counts[i] else None) for i in range(target.shape[1])]
            group_rows[name] = row
        flow = [v for k, v in group_rows.items() if k.startswith('flow:')]
        if sum(v['count'] for v in flow) != int(mask.sum()) or not np.isclose(
                sum(v['global_mae_contribution'] for v in flow), group_rows['all_nodes']['mae_gain_pooled'], atol=1e-10):
            raise ValueError('Flow count/contribution conservation failed')
        r, c = metrics[reference], metrics[candidate]
        comparisons[candidate+'_vs_'+reference] = dict(reference=reference, candidate=candidate, groups=group_rows,
            rmse_gain_pooled=r['rmse_pooled']-c['rmse_pooled'], rmse_gain_macro=None if r['rmse_macro'] is None or c['rmse_macro'] is None else r['rmse_macro']-c['rmse_macro'],
            mape_gain_fraction=r['mape_pooled']-c['mape_pooled'], fixed_hard_window_gain=None if r['fixed_hard_window_mae'] is None or c['fixed_hard_window_mae'] is None else r['fixed_hard_window_mae']-c['fixed_hard_window_mae'])
    return dict(metrics=metrics, comparisons=comparisons, mask_value=null, valid_points=int(mask.sum()),
        fixed_hard_ids=list(hard_ids), bias_reference=bias_case, response_tolerance=response_tol,
        gain_sign='reference error minus candidate error; positive means candidate better',
        mape_unit='fraction', true_flow_and_bias='posthoc only; never prediction inputs',
        interpretation='exploratory checkpoint sensitivity; not retrained deletion or causal contribution')


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parameter_changes(state, source, initial_extra=None, backbone_parameter_names=None):
    backbone = {k.replace('.dif_layer.original.', '.dif_layer.'): v for k, v in state.items() if '.dif_layer.delays.' not in k}
    if set(backbone) != set(source):
        raise ValueError('Cannot match original backbone tensors')
    squared, maximum, positions = 0., 0., 0
    for key in source:
        value = backbone[key].to(torch.float64)-source[key].to(torch.float64)
        squared += float(value.square().sum())
        maximum = max(maximum, float(value.abs().max()))
        positions += value.numel()
    result = dict(backbone_tensor_positions=positions, backbone_change_l2=math.sqrt(squared),
        backbone_change_abs_max=maximum, scope='all state tensors including buffers; not importance')
    if backbone_parameter_names is not None:
        names = set(backbone_parameter_names)
        if not names.issubset(source):
            raise ValueError('Backbone parameter names differ from original state')
        totals = dict(parameter=0., buffer=0.)
        maximum, positions = 0., 0
        for key in source:
            value = backbone[key].to(torch.float64)-source[key].to(torch.float64)
            totals['parameter' if key in names else 'buffer'] += float(value.square().sum())
            if key in names:
                maximum = max(maximum, float(value.abs().max()))
                positions += value.numel()
        result.update(backbone_parameter_positions=positions, backbone_parameter_change_l2=math.sqrt(totals['parameter']),
            backbone_parameter_change_abs_max=maximum, backbone_buffer_change_l2=math.sqrt(totals['buffer']))
    if initial_extra is not None:
        squared, maximum, positions = 0., 0., 0
        for key, initial in initial_extra.items():
            value = state[key].to(torch.float64)-initial.to(torch.float64)
            squared += float(value.square().sum())
            maximum = max(maximum, float(value.abs().max()))
            positions += value.numel()
        result.update(new_parameter_positions=positions, new_parameter_change_l2=math.sqrt(squared),
            new_parameter_change_abs_max=maximum,
            new_parameter_reference='deterministic installed initialization with original pilot seed')
    return result
