"""NumPy-only descriptions of completed GW11 LAST adapter removal.

This module never imports Torch, fits a correction, chooses a winner, modifies
the failed screen, or authorizes more training. OFF means the complete trained
``model.source`` evaluated with that arm's fixed LAST source tensors. It is not
a source that was trained independently without an adapter.
"""
import numpy as np

from experiments.conditional_residual_gw10.screen import metrics as pooled_metrics


ARMS = ('SUM_CONTINUE', 'STATIC_GATE', 'STATE_READOUT', 'STATE_GATE')
GROUPS = ('all', 'high_value_ge314', 'long_horizon_7_12', 'flow_50_200')
METRIC_FIELDS = ('mae', 'rmse', 'mape_fraction', 'signed_bias')
SCHEMA = 'gw11_last_ablation_analysis_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _prediction(value, shape, name):
    value = np.asarray(value)
    _require(value.shape == shape and value.dtype == np.float32
        and np.isfinite(value).all(), 'Finite float32 prediction axes required: '+name)
    return value


def _validate(archive, off_predictions):
    required = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support',
        'pred__FROZEN_SUM', *('pred__'+arm for arm in ARMS))
    _require(all(name in archive for name in required), 'Missing GW11 LAST archive fields')
    y, valid = np.asarray(archive['y']), np.asarray(archive['valid'])
    _require(y.ndim == 3 and y.shape[1] == 12 and min(y.shape) > 0
        and y.dtype == np.float32 and np.isfinite(y).all(),
        'Finite float32 [sample,12,node] truth required')
    _require(valid.shape == y.shape and valid.dtype == np.bool_ and valid.any(),
        'Original nonempty Boolean truth mask required')
    _require((y[valid] > 0.).all(), 'Original valid MAPE targets must be positive')
    for name, count in (('sample_ids', y.shape[0]), ('node_ids', y.shape[2])):
        ids = np.asarray(archive[name])
        _require(ids.shape == (count,) and ids.dtype.kind in 'iu'
            and np.unique(ids).size == count, 'Unique integer axis required: '+name)
    roads, support = np.asarray(archive['road_labels']), np.asarray(archive['direct_support'])
    _require(roads.shape == (y.shape[2],) and roads.dtype.kind in 'US', 'Road axis differs')
    _require(support.shape == (y.shape[0], y.shape[2]) and support.dtype == np.bool_,
        'Direct-support axis differs')
    if 'axis_order' in archive:
        axis = np.asarray(archive['axis_order'])
        _require(axis.ndim == 0 and axis.dtype.kind in 'US'
            and str(axis.item()) == 'sample,horizon,node', 'Explicit archive axis order differs')
    if 'registered_arms' in archive:
        _require(np.array_equal(np.asarray(archive['registered_arms']), np.asarray(ARMS)),
            'Registered GW11 arm order differs')
    _require(isinstance(off_predictions, dict) and set(off_predictions) == set(ARMS),
        'OFF predictions must contain exactly the four predefined arms')
    frozen = _prediction(archive['pred__FROZEN_SUM'], y.shape, 'FROZEN_SUM')
    on = {arm:_prediction(archive['pred__'+arm], y.shape, arm+'__ON') for arm in ARMS}
    off = {arm:_prediction(off_predictions[arm], y.shape, arm+'__OFF') for arm in ARMS}
    _require(np.array_equal(on['SUM_CONTINUE'], off['SUM_CONTINUE']),
        'SUM_CONTINUE ON/OFF must replay exactly; it has no added adapter')
    return y, valid, frozen, on, off


def _metrics(prediction, truth, valid):
    if not valid.any():
        return dict(valid_count=0, **dict.fromkeys(METRIC_FIELDS))
    return pooled_metrics(prediction, truth, valid)


def _group_masks(y):
    return dict(all=np.ones(y.shape, dtype=np.bool_), high_value_ge314=y >= 314.,
        long_horizon_7_12=np.broadcast_to(np.arange(12)[None,:,None] >= 6, y.shape),
        flow_50_200=(y >= 50.) & (y < 200.))


def _record(prediction, y, valid, masks):
    result = _metrics(prediction, y, valid)
    result['groups'] = {name:_metrics(prediction, y, valid & masks[name])
        for name in GROUPS if name != 'all'}
    counts = valid.sum(axis=(1, 2))
    has_values = counts > 0
    absolute = np.abs(prediction.astype(np.float64)-y.astype(np.float64))
    window_mae = np.divide(np.where(valid, absolute, 0.).sum(axis=(1, 2)), counts,
        out=np.zeros(len(counts), dtype=np.float64), where=has_values)
    result['window_p95'] = (float(np.quantile(window_mae[has_values], .95))
        if has_values.any() else None)
    return result


def _effect(after, before, valid):
    delta = after[valid].astype(np.float64)-before[valid].astype(np.float64)
    if not delta.size:
        return dict.fromkeys(('mean', 'mean_absolute', 'rms', 'centered_std'))
    mean = float(delta.mean())
    return dict(mean=mean, mean_absolute=float(np.abs(delta).mean()),
        rms=float(np.sqrt(np.mean(delta*delta))),
        centered_std=float(np.sqrt(np.mean((delta-mean)**2))))


def _close(actual, expected, name, scale):
    # Float32 values are promoted before subtraction and all aggregation. The
    # scale accommodates cancellation when subtracting two large pooled means.
    _require(abs(actual-expected) <= 1e-10*max(1., scale),
        'Paired decomposition identity failed: '+name)


def _decompose(arm, frozen, on, off, truth, valid, total, **details):
    original = _metrics(frozen, truth, valid)
    source = _metrics(off, truth, valid)
    complete = _metrics(on, truth, valid)
    count = original['valid_count']
    row = dict(arm=arm, **details, valid_count=count,
        valid_fraction_of_all=count/total,
        frozen_mae=original['mae'], source_only_mae=source['mae'], complete_mae=complete['mae'],
        frozen_signed_bias=original['signed_bias'], source_only_signed_bias=source['signed_bias'],
        complete_signed_bias=complete['signed_bias'])
    for short, metric in (('mae', 'mae'), ('bias', 'signed_bias')):
        deltas = dict(source_update=None, adapter_presence=None, total=None)
        if count:
            deltas = dict(source_update=source[metric]-original[metric],
                adapter_presence=complete[metric]-source[metric],
                total=complete[metric]-original[metric])
            _close(deltas['source_update']+deltas['adapter_presence'], deltas['total'],
                arm+'/'+short+'/telescoping', max(abs(row_value) for row_value
                    in (source[metric], original[metric], complete[metric])))
        row.update({name+'_'+short+'_delta':value for name, value in deltas.items()})
        row.update({name+'_'+short+'_global_contribution':
            (value*count/total if value is not None else None)
            for name, value in deltas.items()})
    for name, after, before in (('source_update', off, frozen),
            ('adapter_presence', on, off), ('total', on, frozen)):
        effect = _effect(after, before, valid)
        row.update({name+'_effect_'+key:value for key, value in effect.items()})
        if count:
            _close(effect['mean'], row[name+'_bias_delta'], arm+'/'+name+'/mean_equals_bias_delta',
                max(abs(row['frozen_signed_bias']), abs(row['source_only_signed_bias']),
                    abs(row['complete_signed_bias'])))
    row['telescoping_checked'] = bool(count)
    return row


def analyze(archive, off_predictions):
    """Describe paired LAST ON/OFF arrays using the unchanged supplied mask.

    Positive MAE deltas mean worse error. Truth-defined groups are evaluation
    only. No threshold labels an effect as an overall shift or as local action;
    effect means, centered standard deviations, horizons and nodes expose both.
    All rows are pooled over their original valid points, including unequal
    node/window counts. Empty groups have count 0 and unavailable values None.
    """
    y, valid, frozen, on, off = _validate(archive, off_predictions)
    total = int(valid.sum())
    masks = _group_masks(y)
    predictions = {'FROZEN_SUM':frozen}
    for arm in ARMS:
        predictions[arm+'__ON'], predictions[arm+'__OFF'] = on[arm], off[arm]
    records = {name:_record(prediction, y, valid, masks) for name, prediction in predictions.items()}
    result = dict(schema=SCHEMA, status='GW11_LAST_ABLATION_ANALYSIS_COMPLETE',
        selection='last', registered_arms=list(ARMS), metrics=records,
        decomposition=[], by_horizon=[], by_node=[], by_flow=[], by_window=[])

    def add(table, group, index=None, mask=None, **details):
        if index is None:
            truth, selected, reference = y, valid, frozen
        else:
            truth, selected, reference = y[index], valid[index], frozen[index]
        if mask is not None:
            selected = selected & np.broadcast_to(mask, selected.shape)
        for arm in ARMS:
            complete = on[arm] if index is None else on[arm][index]
            source = off[arm] if index is None else off[arm][index]
            result[table].append(_decompose(arm, reference, complete, source,
                truth, selected, total, group=group, **details))

    for name in GROUPS:
        add('decomposition', name, mask=masks[name],
            group_input=('all_original_valid_points' if name == 'all' else
                'forecast_lead_evaluation_only' if name == 'long_horizon_7_12' else
                'future_truth_evaluation_only'))
    for h in range(12):
        add('by_horizon', str(h+1), index=(slice(None), slice(h, h+1), slice(None)), horizon=h+1)
    for n, node in enumerate(archive['node_ids']):
        add('by_node', str(int(node)), index=(slice(None), slice(None), slice(n, n+1)),
            node_id=int(node), road_label=str(archive['road_labels'][n]))
    flow = (('<50', y < 50.), ('50_200', (y >= 50.) & (y < 200.)),
        ('200_314', (y >= 200.) & (y < 314.)), ('ge314', y >= 314.))
    for label, mask in flow:
        add('by_flow', label, mask=mask, group_input='future_truth_evaluation_only')
    for s, sample in enumerate(archive['sample_ids']):
        add('by_window', str(int(sample)), index=(slice(s, s+1), slice(None), slice(None)),
            sample_id=int(sample), order='original_author_sample_order_calendar_not_reverified')
    result['interpretation'] = dict(
        source_update='OFF minus FROZEN_SUM contains all backbone and original SUM-head tensor changes.',
        adapter_presence='ON minus OFF is the direct adapter-presence effect at fixed LAST source tensors.',
        limitation='The source and adapter coadapted during joint training; removing the adapter does not establish the causal effect of training it.',
        effect_scale='Effects are prediction differences in original traffic units; positive MAE delta means worse error.',
        effect_description='Means and centered standard deviations describe constant and heterogeneous components without a selected threshold.',
        groups='Future truth defines evaluation groups only and is never an input to a correction or router.',
        aggregation='Float64 point-pooled means on the original shared Boolean mask; empty groups use null.',
        winner_selection=False, causal_training_claim=False, screening_changed=False,
        screening_failure_does_not_authorize_retuning=True, additional_training_updates=0,
        test_evaluated=False)
    return result


def render_report(result):
    """Render global diagnostics without promoting an arm or choosing a recipe."""
    _require(result.get('status') == 'GW11_LAST_ABLATION_ANALYSIS_COMPLETE',
        'Complete LAST ablation analysis required')
    lines = ['# GW11 LAST fixed-source adapter removal', '',
        'No training or test evaluation. Positive MAE deltas mean worse error.', '',
        '| Arm | Frozen MAE | OFF MAE | ON MAE | Source update delta | Adapter presence delta | ON-OFF mean | ON-OFF centered SD |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['decomposition']:
        if row['group'] != 'all':
            continue
        numbers = [row[name] for name in ('frozen_mae', 'source_only_mae', 'complete_mae',
            'source_update_mae_delta', 'adapter_presence_mae_delta',
            'adapter_presence_effect_mean', 'adapter_presence_effect_centered_std')]
        lines.append('| '+row['arm']+' | '+' | '.join(f'{value:.9f}' for value in numbers)+' |')
    lines += ['', '| Arm | Frozen bias | OFF bias | ON bias | Source bias delta | Adapter bias delta |',
        '|---|---:|---:|---:|---:|---:|']
    for row in result['decomposition']:
        if row['group'] != 'all':
            continue
        numbers = [row[name] for name in ('frozen_signed_bias', 'source_only_signed_bias',
            'complete_signed_bias', 'source_update_bias_delta', 'adapter_presence_bias_delta')]
        lines.append('| '+row['arm']+' | '+' | '.join(f'{value:.9f}' for value in numbers)+' |')
    lines += ['', 'OFF uses each arm\'s jointly trained LAST backbone and original SUM head.',
        'ON versus OFF measures adapter presence with those source tensors fixed. The two parts coadapted;',
        'OFF versus FROZEN_SUM does not isolate the causal effect of the training procedure.',
        'Horizon, node, flow and window tables use the same original valid mask. Truth-defined flow groups',
        'are evaluation only. Means and centered SDs describe movement without declaring an overall shift.',
        'The original failed screen remains unchanged. This analysis selects no winner and authorizes no extra training.', '']
    return '\n'.join(lines)
