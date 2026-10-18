"""GW5: history-anchored recursive logit residuals, with no new parameters.

Only the NEW GW4 correction is constrained. Old GW2 scores still read current
messages; the original graphs are constructed from history once per forward
and reused by native recursion. Call index is not horizon.
"""
from contextlib import contextmanager

import torch
from torch import nn

from experiments.chronological.original_receiver_residual import (
    LogitResidual, ReceiverResidualConv, ReceiverResidualIGSTGNN)


ARMS = ('HISTORY_FIXED', 'HISTORY_ANCHORED', 'PHASE_SHRINK', 'BIAS_SHRINK')
DESIGN = dict(version='GW5', selected_layer=0, residual_family='GW4_RECEIVER_RESIDUAL',
    history_anchor='last_transformed_history_window_residual_logits_per_sample_source_node_slot',
    recursive_generated_fractions=[1./3, 2./3, 1.], recursive_lambdas=[2./3, 1./3, 0.],
    HISTORY_FIXED='anchor', HISTORY_ANCHORED='anchor_plus_lambda_times_free_minus_anchor',
    PHASE_SHRINK='lambda_times_free', BIAS_SHRINK='bias_plus_lambda_times_free_minus_bias',
    history_policy='unchanged_GW4',
    anchor_autograd='retained', anchor_lifetime='one_complete_model_forward',
    extra_trainable_parameters_over_GW4=0, old_parameters_frozen=True, old_modules_eval=True,
    old_GW2_scores='computed_from_current_candidate_messages',
    native_graphs='constructed_once_from_conditioned_history_then_reused_in_recursion',
    separate_report_inputs=False, future_targets_as_inputs=False,
    expected_calls=4, native_history_steps=12, native_kernel=3,
    physical_delay_or_calibrated_confidence_claim=False)


class HistoryLogitResidual(LogitResidual):
    def __init__(self, residual, policy):
        nn.Module.__init__(self)
        if policy not in ARMS or residual.arm != 'RECEIVER_RESIDUAL':
            raise ValueError('GW5 needs receiver residual parameters and a declared policy')
        self.arm, self.hidden = residual.arm, residual.hidden
        self.bias, self.query, self.key = residual.bias, residual.query, residual.key
        self.policy, self.step, self._active = policy, None, False
        self._anchors, self._policy_stats = {}, None

    def begin_forward(self):
        if self._active:
            raise RuntimeError('Overlapping GW5 forwards on one instance are unsupported')
        self._active, self.step, self._anchors = True, None, {}

    def end_forward(self):
        self._active, self.step, self._anchors = False, None, {}

    def forward(self, message, receiver, source, setting=None):
        if not self._active or self.step not in (0, 1, 2, 3):
            raise ValueError('GW5 residual requires one history call followed by at most three recursive calls')
        free, query, key = super().forward(message, receiver, source)
        step = self.step
        rho, mixing = step/3., 1.-step/3.
        policy = 'FREE' if setting == 'FREE' else self.policy
        if step == 0:
            if source in self._anchors:
                raise ValueError('A graph source was scored twice in the history call')
            # Keep autograd; a new forward gets a fresh dictionary, never an
            # in-place replacement of a tensor retained by an earlier graph.
            self._anchors[source] = free[:, -1:].clone()
            effective, anchor = free, None
        else:
            if source not in self._anchors:
                raise ValueError('Missing current-sample history anchor')
            anchor = self._anchors[source]
            if anchor.shape[0] != free.shape[0] or anchor.shape[2:] != free.shape[2:] or free.shape[1] != 1:
                raise ValueError('Native recursive/history sample, node or slot axes changed')
            if setting == 'ANCHOR_ROLL':
                anchor = anchor.roll(1, dims=2)
            if policy == 'FREE':
                effective = free  # Exact GW4 expression, no cancellation.
            elif policy == 'HISTORY_FIXED' or (policy == 'HISTORY_ANCHORED' and mixing == 0.):
                effective = anchor.expand_as(free)
            elif policy == 'HISTORY_ANCHORED':
                effective = anchor+mixing*(free-anchor)
            elif policy == 'PHASE_SHRINK':
                effective = mixing*free
            elif policy == 'BIAS_SHRINK':
                bias = self.bias[source].expand_as(free)
                effective = bias if mixing == 0. else bias+mixing*(free-bias)
            else:
                raise ValueError('Unknown GW5 policy')
        if self._policy_stats is not None:
            self._record_policy(free, effective, anchor, source, step, rho, mixing)
        return effective, query, key

    def _record_policy(self, free, effective, anchor, source, step, rho, mixing):
        tag = (step, source)
        row = self._policy_stats.setdefault(tag, dict(count=0, calls=0, free_abs_sum=0.,
            effective_abs_sum=0., modification_abs_sum=0., departure_abs_sum=0.,
            effective_departure_abs_sum=0., generated_input_fraction=rho,
            scheduled_lambda=mixing))
        f, e = free.detach().double(), effective.detach().double()
        row['count'] += f.numel()
        row['calls'] += 1
        row['free_abs_sum'] += float(f.abs().sum())
        row['effective_abs_sum'] += float(e.abs().sum())
        row['modification_abs_sum'] += float((e-f).abs().sum())
        if anchor is not None:
            a = anchor.detach().double()
            row['departure_abs_sum'] += float((f-a).abs().sum())
            row['effective_departure_abs_sum'] += float((e-a).abs().sum())


class HistoryAnchorConv(ReceiverResidualConv):
    def __init__(self, conv, policy):
        nn.Module.__init__(self)
        self.native, self.router = conv.native, conv.router
        self.mode, self.directions, self.source_labels = conv.mode, conv.directions, conv.source_labels
        self.arm, self.override_mode, self.call_index = policy, None, 0
        self._stats, self._residual_stats = None, None
        self.residual = HistoryLogitResidual(conv.residual, policy)

    def forward(self, x, dynamic_graph, static_graph):
        if not self.residual._active:
            raise RuntimeError('Call the GW5 model wrapper, not its convolution in isolation')
        if self.call_index >= DESIGN['expected_calls'] or x.shape[1] != (12 if self.call_index == 0 else 3):
            raise ValueError('Frozen native DIF history/recursive call contract changed')
        return super().forward(x, dynamic_graph, static_graph)

    def weights(self, message, receiver, source):
        # The inherited convolution increments its call index before scoring.
        self.residual.step = self.call_index-1
        return super().weights(message, receiver, source)

    def start_diagnostics(self):
        super().start_diagnostics()
        self.residual._policy_stats = {}

    def finish_diagnostics(self):
        result = super().finish_diagnostics()
        raw, self.residual._policy_stats = self.residual._policy_stats or {}, None
        result.update(version='GW5', policy=self.arm, policy_by_phase={},
            phase_interpretation='pre_mixing_input_provenance_counts_not_confidence_or_true_horizon')
        for (step, source), row in raw.items():
            phase = 'history' if step == 0 else 'forecast_'+str(step)
            result['policy_by_phase'].setdefault(phase, {})[self.source_labels[source]] = dict(
                calls=row['calls'], generated_input_fraction=row['generated_input_fraction'],
                scheduled_lambda=row['scheduled_lambda'],
                free_logit_abs_mean=row['free_abs_sum']/row['count'],
                effective_logit_abs_mean=row['effective_abs_sum']/row['count'],
                policy_modification_abs_mean=row['modification_abs_sum']/row['count'],
                free_departure_from_history_abs_mean=row['departure_abs_sum']/row['count'] if step else None,
                effective_departure_from_history_abs_mean=row['effective_departure_abs_sum']/row['count'] if step else None)
        return result


class HistoryAnchorIGSTGNN(ReceiverResidualIGSTGNN):
    def __init__(self, anchor, policy):
        if policy not in ARMS:
            raise ValueError('Unknown GW5 training policy')
        super().__init__(anchor, 'RECEIVER_RESIDUAL')
        if self.seq_len != 12 or self.horizon != 12 or self.selector.native.k_t != 3:
            raise ValueError('GW5 is scoped to the authenticated native 12/12/K3 contract')
        self.base.original.layers[0].dif_layer.localized_st_conv = HistoryAnchorConv(self.selector, policy)
        self.arm = policy
        self.train(False)

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        residual = self.selector.residual
        residual.begin_forward()
        try:
            result = super().forward(history_data, incident_data=incident_data, sensor_data=sensor_data)
            if self.selector.call_index != DESIGN['expected_calls']:
                raise ValueError('Native DIF did not make its authenticated four calls')
            return result
        finally:
            residual.end_forward()

    @contextmanager
    def intervention(self, setting):
        if setting not in ('BASE', 'FREE', 'ANCHOR_ROLL') or (
                setting == 'ANCHOR_ROLL' and self.arm != 'HISTORY_ANCHORED'):
            raise ValueError('Unknown or inapplicable GW5 intervention')
        previous = self.selector.override_mode
        self.selector.override_mode = setting
        try:
            yield self
        finally:
            self.selector.override_mode = previous
