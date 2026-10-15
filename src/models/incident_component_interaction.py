"""Standalone, zero-start score interactions for a possible F2 extension.

This module does not fuse components, predict traffic, run a backbone, or add
state/graph propagation. Its outputs are uncentered score residuals. A caller
should center each new branch across the ten components before adding it to
existing scores and applying the existing centering/bounding policy once.
All flow inputs must use the same existing training normalization. In
particular, ``fixed_sum_prediction`` is a differentiable, complete prediction
from the fixed component sum; it is neither an individual contribution nor a
future label. The module cannot verify a tensor's data provenance.
"""

import torch
from torch import nn


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class InteractionScoreResidual(nn.Module):
    """Independent history and report score branches with fixed rank eight.

    History evidence is [last, mean, last-first, prediction-last,
    prediction-mean], concatenated with four horizon features. Each branch
    has its own component/history factors. The report branch additionally
    multiplies a bias-free report factor and an external Boolean support mask.
    Only the two bias-free final readout weights start at zero. Other factors
    use PyTorch's random Linear initialization. Bounded tanh factors avoid
    arbitrarily large multiplicative features; the scores are not bounded.

    ``history_enabled=False`` disables the history *score* branch only. The
    independent report interaction still reads history evidence. Disabling
    reports ignores report arguments entirely. With reports enabled, both
    report arguments may be None to represent missing reports; supplied report
    features require an explicit support mask. There is no implicit support
    inference from feature values.
    """

    rank = 8
    horizon = 12
    gap = 3
    components = 10
    report_width = 19
    history_feature_names = ('last', 'mean', 'last_minus_first',
                             'prediction_minus_last', 'prediction_minus_mean')

    def __init__(self, component_dim, *, rank=8, horizon_dim=4, report_dim=19):
        super().__init__()
        _require(type(component_dim) is int and component_dim > 0,
                 'component_dim must be a positive integer')
        _require(type(rank) is int and rank == self.rank, 'interaction requires rank=8')
        _require(type(horizon_dim) is int and horizon_dim == 4,
                 'interaction requires horizon_dim=4')
        _require(type(report_dim) is int and report_dim == self.report_width,
                 'interaction requires report_dim=19')
        self.component_dim = component_dim
        self.history_component = nn.Linear(component_dim, self.rank)
        self.history_projection = nn.Linear(5+4, self.rank)
        self.history_readout = nn.Linear(self.rank, 1, bias=False)
        self.report_component = nn.Linear(component_dim, self.rank)
        self.report_history_projection = nn.Linear(5+4, self.rank)
        self.report_projection = nn.Linear(self.report_width, self.rank, bias=False)
        self.report_readout = nn.Linear(self.rank, 1, bias=False)
        nn.init.zeros_(self.history_readout.weight)
        nn.init.zeros_(self.report_readout.weight)

    @staticmethod
    def _validate_tensor(name, value, shape, reference):
        _require(isinstance(value, torch.Tensor), f'{name} must be a Tensor')
        _require(tuple(value.shape) == tuple(shape), f'{name} shape must be {tuple(shape)}')
        _require(value.is_floating_point(), f'{name} must have a floating dtype')
        _require(value.dtype == reference.dtype and value.device == reference.device,
                 f'{name} dtype/device must match component_encodings')
        _require(bool(torch.isfinite(value).all()), f'{name} must be finite')

    def _component_scores(self, encodings, context_factor, projection, readout):
        """Contract compact blocks with per-slot context without a 10x rank stack."""
        batch, blocks, nodes, _, _ = encodings.shape
        grouped_context = context_factor.reshape(batch, blocks, self.gap, nodes, self.rank)
        scores = []
        for index in range(self.components):
            factor = projection(encodings[..., index, :]).tanh()
            weighted_factor = factor*readout.weight.reshape(1, 1, 1, self.rank)
            scores.append(torch.einsum('bknr,bkgnr->bkgn', weighted_factor,
                                       grouped_context).reshape(batch, self.horizon, nodes))
        return torch.stack(scores, dim=-1)

    def forward(self, component_encodings, historical_flow, fixed_sum_prediction,
                horizon_features, report_features=None, support=None, *, gap=3,
                history_enabled=True, report_enabled=True, collect=False):
        """Return named [B,12,N,10] scores and optional detached diagnostics.

        Compact encodings have shape [B,4,N,10,C]; history and complete
        predictions have shape [B,12,N,1]. Horizon features are [12,4].
        Reports/support, when present, are [B,N,19]/Bool[B,N,1].
        No inputs or live scores are detached. Diagnostics alone are detached.
        """
        _require(all(type(flag) is bool for flag in
                     (history_enabled, report_enabled, collect)),
                 'interaction switches must be bool')
        _require(type(gap) is int and gap == self.gap, 'interaction requires gap=3')
        _require(isinstance(component_encodings, torch.Tensor),
                 'component_encodings must be a Tensor')
        shape = component_encodings.shape
        _require(len(shape) == 5 and shape[0] > 0 and shape[1] == self.horizon//gap
                 and shape[2] > 0 and shape[3:] == (self.components, self.component_dim),
                 'component_encodings shape must be [B,4,N,10,component_dim]')
        batch, _, nodes, _, _ = shape
        reference = component_encodings
        self._validate_tensor('component_encodings', component_encodings, shape, reference)
        _require(reference.dtype == self.history_readout.weight.dtype
                 and reference.device == self.history_readout.weight.device,
                 'module dtype/device must match component_encodings')
        self._validate_tensor('historical_flow', historical_flow,
                              (batch, self.horizon, nodes, 1), reference)
        self._validate_tensor('fixed_sum_prediction', fixed_sum_prediction,
                              (batch, self.horizon, nodes, 1), reference)
        self._validate_tensor('horizon_features', horizon_features,
                              (self.horizon, 4), reference)

        report_present = report_enabled and report_features is not None
        if report_enabled:
            if report_present:
                self._validate_tensor('report_features', report_features,
                                      (batch, nodes, self.report_width), reference)
                _require(isinstance(support, torch.Tensor),
                         'supplied report_features require an explicit support Tensor')
            if support is not None:
                _require(isinstance(support, torch.Tensor)
                         and support.shape == (batch, nodes, 1),
                         'support shape must be [B,N,1]')
                _require(support.dtype == torch.bool and support.device == reference.device,
                         'support must be Bool on the component_encodings device')
                _require(report_present or not bool(support.any()),
                         'support cannot be true when report_features are missing')

        last = historical_flow[:, -1:, :, :].expand(-1, self.horizon, -1, -1)
        mean = historical_flow.mean(1, keepdim=True).expand(-1, self.horizon, -1, -1)
        change = (historical_flow[:, -1:, :, :]-historical_flow[:, :1, :, :]).expand_as(last)
        history_features = torch.cat((last, mean, change,
                                      fixed_sum_prediction-last,
                                      fixed_sum_prediction-mean), dim=-1)
        _require(bool(torch.isfinite(history_features).all()),
                 'derived history_features must be finite')
        position = horizon_features[None, :, None, :].expand(batch, -1, nodes, -1)
        context = torch.cat((history_features, position), dim=-1)
        zero = reference.new_zeros(batch, self.horizon, nodes, self.components)

        history_scores = zero
        if history_enabled:
            history_factor = self.history_projection(context).tanh()
            history_scores = self._component_scores(component_encodings, history_factor,
                                                     self.history_component, self.history_readout)

        report_scores = zero
        if report_present:
            history_factor = self.report_history_projection(context).tanh()
            report_factor = self.report_projection(report_features).tanh()[:, None, :, :]
            report_scores = self._component_scores(component_encodings,
                history_factor*report_factor, self.report_component, self.report_readout)
            report_scores = report_scores*support[:, None, :, :].to(reference.dtype)

        _require(bool(torch.isfinite(history_scores).all())
                 and bool(torch.isfinite(report_scores).all()), 'interaction scores must be finite')
        diagnostics = None
        if collect:
            with torch.no_grad():
                history = history_scores.detach()
                report = report_scores.detach()
                diagnostics = dict(
                    rank=self.rank, gap=gap, history_enabled=history_enabled,
                    report_enabled=report_enabled, report_present=report_present,
                    history_feature_names=self.history_feature_names,
                    history_features=history_features.detach(),
                    supported_node_positions=(support.sum().detach() if report_present
                                              else reference.new_zeros((), dtype=torch.long)),
                    history_score_abs_mean_by_component=history.abs().mean((0, 1, 2)),
                    report_score_abs_mean_by_component=report.abs().mean((0, 1, 2)),
                    history_centered_score_abs_max=(history-history.mean(-1, keepdim=True)).abs().max(),
                    report_centered_score_abs_max=(report-report.mean(-1, keepdim=True)).abs().max())
        return dict(history_scores=history_scores, report_scores=report_scores,
                    diagnostics=diagnostics)
