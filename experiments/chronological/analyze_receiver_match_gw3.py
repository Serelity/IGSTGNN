"""Read-only GW3/GW2 diagnosis from NPZ/JSON; NumPy and stdlib only.

No torch/model imports, raw data, checkpoint deserialization, new inference,
optimizer updates or test evaluation. Missing evidence is never replaced by a
model run. Saved mechanism pointers may be followed inside the declared runs.
"""
import argparse
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCHEMA = 'gw3_read_only_diagnosis_v1'
GW3_SCHEMA = 'original_igstgnn_receiver_match_archive_v1'
GW2_SCHEMA = 'original_igstgnn_graph_message_reuse_archive_v1'
OLD_ARMS = ('ORIGINAL', 'STATIC', 'POST_MESSAGE')
NEW_ARMS = ('FIXED_UNIFORM', 'MESSAGE_SELF', 'RECEIVER_MATCH')
ARMS = (*OLD_ARMS, *NEW_ARMS)
GW3_CASES = (*ARMS, 'MESSAGE_SELF_UNIFORM', 'MESSAGE_SELF_STATIC_ONLY',
    'RECEIVER_MATCH_UNIFORM', 'RECEIVER_MATCH_STATIC_ONLY', 'RECEIVER_MATCH_QUERY_SELF')
GW2_CASES = ('ORIGINAL', 'STATIC', 'POST_LOCAL', 'POST_MESSAGE', 'POST_LOCAL_UNIFORM',
    'POST_MESSAGE_UNIFORM', 'POST_MESSAGE_STATIC_ONLY')
COMPARISONS = (
    ('RECEIVER_MATCH_vs_POST_MESSAGE', 'POST_MESSAGE', 'RECEIVER_MATCH'),
    ('RECEIVER_MATCH_vs_ORIGINAL', 'ORIGINAL', 'RECEIVER_MATCH'),
    ('RECEIVER_MATCH_vs_STATIC', 'STATIC', 'RECEIVER_MATCH'),
    ('RECEIVER_MATCH_vs_FIXED_UNIFORM', 'FIXED_UNIFORM', 'RECEIVER_MATCH'),
    ('RECEIVER_MATCH_vs_MESSAGE_SELF', 'MESSAGE_SELF', 'RECEIVER_MATCH'),
    ('receiver_selector_dependency', 'RECEIVER_MATCH_UNIFORM', 'RECEIVER_MATCH'),
    ('receiver_content_dependency', 'RECEIVER_MATCH_STATIC_ONLY', 'RECEIVER_MATCH'),
    ('receiver_query_dependency', 'RECEIVER_MATCH_QUERY_SELF', 'RECEIVER_MATCH'),
    ('message_selector_dependency', 'MESSAGE_SELF_UNIFORM', 'MESSAGE_SELF'),
    ('gw2_selector_dependency', 'GW2_POST_MESSAGE_UNIFORM', 'POST_MESSAGE'),
    ('uniform_GW3_vs_GW2', 'GW2_POST_MESSAGE_UNIFORM', 'RECEIVER_MATCH_UNIFORM'))
AXES = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')
PRODUCERS = ('evaluate_original_receiver_match.py', 'receiver_match_reference.py',
    'evaluate_original_graph_message.py', 'evaluate_unified_adapter_v03.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Inputs:
    def __init__(self):
        self.files = {}

    def record(self, path):
        path = Path(path).resolve()
        require(path.suffix in ('.json', '.npz', '.py'), 'Diagnosis reads JSON/NPZ/source only: '+str(path))
        value = dict(sha256=sha256(path), bytes=path.stat().st_size)
        if str(path) in self.files:
            require(self.files[str(path)] == value, 'Source changed during analysis: '+str(path))
        self.files[str(path)] = value
        return value['sha256']

    def json(self, path):
        self.record(path)
        result = json.loads(Path(path).read_text(encoding='utf-8'))
        self.record(path)
        return result

    def npz(self, path, expected):
        require(self.record(path) == expected, 'Prediction archive SHA256 differs: '+str(path))
        with np.load(path, allow_pickle=False) as data:
            result = dict(data)
        self.record(path)
        return result

    def preserved(self):
        require(all(sha256(path) == row['sha256'] for path, row in self.files.items()), 'An input changed during analysis')
        return dict(status='SOURCE_FILES_UNCHANGED', files=self.files, checkpoint_files_read=0,
                    raw_data_files_read=0, model_imports=0, new_inference_calls=0, optimizer_updates=0)


def identity_digest(identity):
    body = {key:value for key, value in identity.items() if key != 'identity_sha256'}
    require(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == identity.get('identity_sha256'),
            'Saved training identity digest differs')


def validate_archive(data, schema, cases):
    require(str(np.asarray(data.get('schema')).item()) == schema, 'Prediction schema differs')
    require(str(np.asarray(data.get('axis_order')).item()) == 'sample,horizon,node', 'Explicit saved axes required')
    y, valid = data['y'], data['valid']
    require(y.ndim == 3 and y.shape[1] == 12 and min(y.shape) > 0 and y.dtype.kind == 'f', 'Expected floating [sample,12,node]')
    require(valid.dtype == np.bool_ and valid.shape == y.shape and valid.any(), 'Saved mask differs')
    require(np.isfinite(y).all() and (y[valid] > 0).all(), 'Invalid positive valid targets')
    null = float(y.min()) if y.min() < 1 else 0.
    require(np.array_equal(valid, y != null), 'Mask differs from original global-null rule')
    for name, length in (('sample_ids', len(y)), ('node_ids', y.shape[2])):
        value = data[name]
        require(value.shape == (length,) and value.dtype.kind in 'iu' and len(np.unique(value)) == length, 'Invalid '+name)
    require(data['road_labels'].shape == (y.shape[2],) and data['road_labels'].dtype.kind in 'US', 'Road labels differ')
    require(data['direct_support'].shape == (len(y), y.shape[2]) and data['direct_support'].dtype == np.bool_, 'Support axes differ')
    predictions = {key[6:]:value for key, value in data.items() if key.startswith('pred__')}
    require(set(predictions) == set(cases), 'Exactly the certified case set is required')
    for name, prediction in predictions.items():
        require(prediction.shape == y.shape and prediction.dtype.kind == 'f' and np.isfinite(prediction).all(), 'Invalid prediction: '+name)
    return predictions


def ratio(total, count):
    return float(total/count) if count else None


def metrics(prediction, y, mask):
    target = y[mask].astype(np.float64)
    error = prediction[mask].astype(np.float64)-target
    count = len(error)
    return dict(valid_count=count, mae=ratio(np.abs(error).sum(), count),
        rmse=None if not count else float(np.sqrt(np.square(error).mean())),
        mape=ratio((np.abs(error)/target).sum(), count), signed_error=ratio(error.sum(), count))


def window_scores(prediction, y, valid):
    counts = valid.sum(2)
    complete = (counts > 0).all(1)
    error = np.where(valid, np.abs(prediction.astype(np.float64)-y), 0.)
    scores = np.full(len(y), np.nan)
    scores[complete] = (error.sum(2)[complete]/counts[complete]).mean(1)
    return scores, complete


def fixed_hard(original, y, valid, ids):
    scores, complete = window_scores(original, y, valid)
    eligible = np.flatnonzero(complete)
    order = np.lexsort((ids[eligible], -scores[eligible]))
    return eligible[order[:math.ceil(.1*len(eligible))]]


def paired(control, candidate, y, mask, global_count):
    a, b = metrics(control, y, mask), metrics(candidate, y, mask)
    c, p, truth = (value[mask].astype(np.float64) for value in (control, candidate, y))
    gain = np.abs(c-truth)-np.abs(p-truth)
    count = len(gain)
    positive, negative = float(np.maximum(gain, 0).sum()), float(np.maximum(-gain, 0).sum())
    return dict(valid_count=count, mae_gain=ratio(gain.sum(), count),
        rmse_gain=None if not count else a['rmse']-b['rmse'],
        mape_gain=None if not count else a['mape']-b['mape'],
        positive_gain=ratio(positive, count), negative_gain=ratio(negative, count),
        cancellation=None if not count else (1.-abs(float(gain.sum()))/(positive+negative) if positive+negative else 0.),
        global_mae_contribution=ratio(gain.sum(), global_count),
        prediction_shift_mean=ratio((p-c).sum(), count), prediction_abs_change_mean=ratio(np.abs(p-c).sum(), count),
        response_fraction=ratio(np.count_nonzero(np.abs(p-c) > .001), count))


def groups(data, hard):
    y = data['y']
    direct = np.broadcast_to(data['direct_support'][:, None], y.shape)
    selected = np.zeros(len(y), bool); selected[hard] = True
    result = dict(all_nodes=np.ones(y.shape, bool), fixed_hard_windows=np.broadcast_to(selected[:, None, None], y.shape),
        direct_report_nodes=direct, outside_direct_support=~direct)
    edges = (0., 5., 20., 50., 100., 200., float('inf'))
    for lo, hi in zip(edges[:-1], edges[1:]):
        result[f'flow:[{lo:g},{hi:g})'] = (y >= lo) & (y < hi)
    for label in np.unique(data['road_labels']):
        result['road:'+str(label)] = np.broadcast_to((data['road_labels'] == label)[None, None], y.shape)
    baseline = data['pred__POST_MESSAGE']
    result.update(bias_over=baseline > y, bias_under=baseline < y, bias_exact=baseline == y)
    return result


def directions(control, candidate, y, valid, global_count):
    c, p, target = (value[valid].astype(np.float64) for value in (control, candidate, y))
    before, after, change = target-c, target-p, p-c
    gain = np.abs(before)-np.abs(after)
    crossing = before*after < 0
    labels = np.full(len(c), 'away_from_target', dtype='<U32')
    labels[change == 0] = 'unchanged'
    labels[(change != 0) & (before == 0)] = 'exact_control_changed'
    labels[(change != 0) & (before != 0) & (after == 0)] = 'reached_target'
    labels[(change != 0) & (gain > 0) & ~crossing & (after != 0)] = 'toward_without_crossing'
    labels[crossing & (gain > 0)] = 'overshoot_improved'
    labels[crossing & (gain == 0)] = 'overshoot_equal_error'
    labels[crossing & (gain < 0)] = 'overshoot_worsened'
    return [dict(direction=str(label), valid_count=int((labels == label).sum()),
        mae_gain=ratio(gain[labels == label].sum(), int((labels == label).sum())),
        global_mae_contribution=ratio(gain[labels == label].sum(), global_count)) for label in np.unique(labels)]


def error_identity(predictions, y, valid, masks):
    error = lambda name: np.abs(predictions[name].astype(np.float64)-y)
    full = error('POST_MESSAGE')-error('RECEIVER_MATCH')
    uniform = error('GW2_POST_MESSAGE_UNIFORM')-error('RECEIVER_MATCH_UNIFORM')
    dep3 = error('RECEIVER_MATCH_UNIFORM')-error('RECEIVER_MATCH')
    dep2 = error('GW2_POST_MESSAGE_UNIFORM')-error('POST_MESSAGE')
    residual = full-uniform-dep3+dep2
    maximum = float(np.abs(residual[valid]).max())
    require(maximum < 1e-9, 'Pointwise absolute-error identity failed')
    rows = []
    for group, group_mask in masks.items():
        mask = valid & group_mask
        count = int(mask.sum())
        rows.append(dict(group=group, valid_count=count,
            full_mae_gain=ratio(full[mask].sum(), count), uniform_mae_gain=ratio(uniform[mask].sum(), count),
            gw3_selector_dependency=ratio(dep3[mask].sum(), count), gw2_selector_dependency=ratio(dep2[mask].sum(), count),
            residual=ratio(residual[mask].sum(), count)))
    return dict(status='PAIRED_ABSOLUTE_ERROR_IDENTITY_PASS', pointwise_residual_abs_max=maximum,
        caveat='descriptive_identity_not_causal_or_independent_training_contributions'), rows


def certificate(meta, identity, summary, selection):
    require(meta.get('schema') == 'gw3_receiver_match_producer_v1' and meta.get('status') == 'GW3_PREDICTION_PASS', 'Completed GW3 producer required')
    require(meta.get('run_identity_sha256') == identity['identity_sha256'], 'Evaluation belongs to another run')
    require(meta.get('engineering_only') is False, 'Use full validation predictions')
    require(meta.get('common_completed_epochs') == summary['common_completed_epochs'], 'Evaluation budget differs')
    for arm in ARMS:
        state, role = summary['arms'][arm], meta.get('checkpoint_roles', {}).get(arm, {})
        epoch = state['best_epoch'] if selection == 'best' else state['completed_epoch']
        require(role.get('selection') == selection and role.get('epoch') == epoch, 'Cached checkpoint role differs: '+arm)
        if selection == 'best':
            require(role.get('checkpoint_sha256') == state['best_checkpoint_sha256'], 'Cached best digest differs: '+arm)
        elif arm in NEW_ARMS:
            require(role.get('model_tensor_sha256') == state['last_model_tensor_sha256'], 'Cached last tensor identity differs: '+arm)
        else:
            require(role.get('checkpoint_sha256') == meta['reference']['reused_arms'][arm]['last_checkpoint_sha256'], 'Old last digest differs: '+arm)
        require(meta.get('validation_replay', {}).get(arm, {}).get('passed') is True, 'Missing saved validation replay: '+arm)


def find_evaluation(run, identity, summary, selection, ledger, explicit=None):
    paths = [Path(explicit).resolve()] if explicit else sorted(path.parent for path in run.glob('*/metadata.json'))
    accepted = []
    for directory in paths:
        try:
            # Non-matching candidates are not part of the selected evidence.
            meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            certificate(meta, identity, summary, selection)
            require(sha256(directory/'paired_predictions.npz') == meta.get('archive_sha256'), 'Archive digest differs')
            accepted.append(directory)
        except (OSError, ValueError, KeyError):
            if explicit:
                raise
    require(bool(accepted), 'No certified '+selection+' evaluation; supply --'+selection+'-eval with its existing directory')
    directory = accepted[-1]
    meta = ledger.json(directory/'metadata.json')
    certificate(meta, identity, summary, selection)
    for name in PRODUCERS:
        source = REPO/'experiments/chronological'/name
        require(ledger.record(source) == meta.get('producer_source', {}).get(name), 'Saved producer source differs: '+name)
    return directory, meta


def load_pair(directory, meta, ledger, selection):
    data = ledger.npz(directory/'paired_predictions.npz', meta['archive_sha256'])
    predictions = validate_archive(data, GW3_SCHEMA, GW3_CASES)
    saved = meta['reference']['archives'][selection]
    old_directory = Path(saved['directory']).resolve()
    require(ledger.record(old_directory/'metadata.json') == saved['metadata_sha256'], 'GW2 metadata digest differs')
    old_meta = ledger.json(old_directory/'metadata.json')
    require(old_meta.get('status') == 'GW2_REUSE_PREDICTION_PASS' and old_meta.get('schema') == 'original_igstgnn_graph_message_reuse_producer_v1', 'Completed GW2 cache required')
    require(old_meta.get('run_identity_sha256') == meta['reference']['source_identity_sha256'], 'GW2 source identity differs')
    require(old_meta.get('common_completed_epochs') == meta['common_completed_epochs'], 'GW2/GW3 budgets differ')
    require(old_meta.get('engineering_only') is False, 'GW2 engineering/full mismatch')
    require(old_meta.get('reuse_reference') == meta['reference']['gw1_reference'], 'GW2 original-source provenance differs')
    for arm in OLD_ARMS:
        require({k:v for k, v in old_meta.get('checkpoint_roles', {}).get(arm, {}).items() if k != 'reused'} == {
            k:v for k, v in meta['checkpoint_roles'][arm].items() if k != 'reused'}, 'GW2 cached role differs: '+arm)
        require(old_meta.get('validation_replay', {}).get(arm, {}).get('passed') is True, 'GW2 saved replay missing: '+arm)
    require(old_meta.get('archive_sha256') == saved['archive_sha256'], 'GW2 archive certificate differs')
    old = ledger.npz(old_directory/'paired_predictions.npz', saved['archive_sha256'])
    old_predictions = validate_archive(old, GW2_SCHEMA, GW2_CASES)
    require(all(np.array_equal(data[key], old[key]) for key in AXES), 'GW2/GW3 target, mask or axis differs')
    for arm in OLD_ARMS:
        require(np.array_equal(predictions[arm], old_predictions[arm]), 'Reused baseline prediction changed: '+arm)
    predictions['GW2_POST_MESSAGE_UNIFORM'] = old_predictions['POST_MESSAGE_UNIFORM']
    data['pred__GW2_POST_MESSAGE_UNIFORM'] = predictions['GW2_POST_MESSAGE_UNIFORM']
    return data, predictions, old_directory


def source_identities(reference, target, ledger):
    definitions = ((reference, 'original_igstgnn_graph_message_training_v1'),
                   (reference['gw1_reference'], 'original_igstgnn_graph_window_training_v1'))
    for saved, schema in definitions:
        path = Path(saved['source_run'])/'identity.json'
        require(ledger.record(path) == saved['source_identity_file_sha256'], 'Historical identity file changed')
        identity = ledger.json(path)
        identity_digest(identity)
        require(identity.get('schema') == schema, 'Historical model family differs')
        require(identity['identity_sha256'] == saved['source_identity_sha256'], 'Historical identity digest differs')
        for field in ('data', 'training_config', 'source_fingerprint', 'engineering_only'):
            require(identity[field] == target[field], 'Historical training definition differs: '+field)
        for field in ('original_tensor_sha256', 'post_construction_rng_sha256'):
            require(identity['initializations']['ORIGINAL'][field] == target['initializations']['ORIGINAL'][field],
                    'Historical native initialization differs: '+field)


def validate_mechanisms(stats):
    sources = stats.get('source_labels')
    require(isinstance(sources, list) and bool(sources) and len(set(sources)) == len(sources), 'No valid saved routing source labels')
    means, std, entropy = (np.asarray(stats.get(key), dtype=np.float64) for key in ('weight_mean', 'weight_std', 'entropy_mean'))
    require(means.ndim == 2 and means.shape[0] == len(sources) and means.shape[1] == 3
            and std.shape == means.shape and entropy.shape == (len(sources),), 'Saved routing aggregate axes differ')
    require(np.isfinite(means).all() and np.isfinite(std).all() and np.isfinite(entropy).all()
            and (std >= 0).all(), 'Invalid saved routing aggregates')
    require(np.allclose(means.sum(-1), 3., atol=1e-5, rtol=1e-5), 'Mean coefficient mass differs')
    phases = stats.get('by_phase')
    require(isinstance(phases, dict) and bool(phases), 'Saved call-phase statistics unavailable')
    for phase, rows in phases.items():
        require(phase == 'history' or phase.startswith('forecast_'), 'Unknown saved call phase')
        require(set(rows) == set(sources), 'Phase source set differs')
        for row in rows.values():
            mean = np.asarray(row.get('weight_mean'), dtype=np.float64)
            variation = np.asarray(row.get('across_sample_std_rms'), dtype=np.float64)
            require(mean.shape == variation.shape == (3,) and np.isfinite(mean).all()
                and np.isfinite(variation).all() and (variation >= 0).all(), 'Phase routing values differ')
            require(row.get('samples', 0) > 0 and row.get('calls', 0) > 0, 'No saved phase observations')


def flatten_mechanisms(stats, **prefix):
    summary, phases = [], []
    sources = stats.get('source_labels', [])
    for i, source in enumerate(sources):
        entropy = stats.get('entropy_mean', [None]*len(sources))[i]
        weights = stats.get('weight_mean', [[]]*len(sources))[i]
        std = stats.get('weight_std', [[]]*len(sources))[i]
        summary.append(dict(prefix, source=source, entropy_mean=entropy,
            normalized_entropy=None if entropy is None or len(weights) < 2 else float(entropy/math.log(len(weights))),
            **{'weight_mean_'+str(j):float(v) for j, v in enumerate(weights)},
            **{'mixed_axis_weight_std_'+str(j):float(v) for j, v in enumerate(std)}))
    for phase, source_rows in stats.get('by_phase', {}).items():
        for source, row in source_rows.items():
            phases.append(dict(prefix, phase=phase, source=source, samples=row.get('samples'),
                calls=row.get('calls'), local_window_positions=row.get('local_window_positions'),
                **{'weight_mean_'+str(j):float(v) for j, v in enumerate(row.get('weight_mean', []))},
                **{'across_sample_std_rms_'+str(j):float(v) for j, v in enumerate(row.get('across_sample_std_rms', []))}))
    return summary, phases


def resolve_mechanisms(directory, case, ledger, allowed_roots, seen=None):
    directory = Path(directory).resolve()
    roots = [Path(root).resolve() for root in allowed_roots]
    require(any(directory == root or root in directory.parents for root in roots), 'Mechanism pointer is outside declared runs')
    seen = set() if seen is None else seen
    key = (str(directory), case)
    require(key not in seen and len(seen) < 12, 'Cyclic or too deep mechanism pointer')
    seen.add(key)
    payload = ledger.json(directory/'mechanisms.json')
    require(case in payload, 'Saved mechanism case missing: '+case)
    stats = payload[case]
    if stats.get('reused'):
        if stats.get('source_directory'):
            next_directory = Path(stats['source_directory'])
        elif stats.get('source_mechanisms_path'):
            source_path = Path(stats['source_mechanisms_path'])
            require(source_path.name == 'mechanisms.json', 'Unexpected mechanism reference filename')
            next_directory = source_path.parent
        else:
            raise ValueError('Reused mechanism has no source pointer')
        result, chain = resolve_mechanisms(next_directory, case, ledger, roots, seen)
        return result, [str(directory), *chain]
    validate_mechanisms(stats)
    return stats, [str(directory)]


def histories(run, identity, summary, reference, ledger):
    roots = {arm:run for arm in NEW_ARMS}
    roots['POST_MESSAGE'] = Path(reference['source_run'])
    roots.update({arm:Path(reference['gw1_reference']['source_run']) for arm in ('ORIGINAL', 'STATIC')})
    rows, mechanism_rows = [], []
    count = len(identity['data']['splits']['train']['partition'])
    for arm, root in roots.items():
        path = root/arm/'history.json'
        history = ledger.json(path)
        saved = summary['arms'][arm]
        require(len(history) == saved['completed_epoch'] and [r['epoch'] for r in history] == list(range(1, len(history)+1)), 'Invalid saved history: '+arm)
        require(all(r['global_updates'] == r['epoch']*count for r in history), 'History update budget differs: '+arm)
        best = min(history, key=lambda r:r['validation_mae'])
        require(best['epoch'] == saved['best_epoch'], 'History best selection differs: '+arm)
        for record, selected in ((best, saved['best_metrics']), (history[-1], saved['last_metrics'])):
            for metric, field in (('mae', 'mae'), ('rmse', 'rmse'), ('mape_fraction', 'mape_fraction')):
                require(np.isclose(record['validation_'+metric], selected[field], atol=1e-7, rtol=1e-7), 'History/summary metric differs: '+arm)
        previous = None
        for row in history:
            values = [row['validation_mae'], row['validation_rmse'], row['validation_mape_fraction']]
            require(np.isfinite(values).all(), 'Nonfinite history: '+arm)
            rows.append(dict(arm=arm, epoch=row['epoch'], global_updates=row['global_updates'],
                train_mae=row.get('train_mae'), validation_mae=values[0], validation_rmse=values[1],
                validation_mape_fraction=values[2], validation_mae_change=None if previous is None else values[0]-previous,
                selected_best=row['epoch'] == saved['best_epoch'], learning_rate=row.get('learning_rate'),
                train_seconds=row.get('train_seconds'), source_history=str(path.resolve())))
            previous = values[0]
            _, phase_rows = flatten_mechanisms(row.get('mechanisms', {}), arm=arm, epoch=row['epoch'])
            mechanism_rows.extend(phase_rows)
    return rows, mechanism_rows


def analyze_pair(selection, data, predictions, output, progress=False, fixed_hard_ids=None):
    y, valid = data['y'], data['valid']
    count = int(valid.sum())
    if fixed_hard_ids is None:
        hard = fixed_hard(predictions['ORIGINAL'], y, valid, data['sample_ids'])
    else:
        lookup = {int(value):i for i, value in enumerate(data['sample_ids'])}
        require(len(set(fixed_hard_ids)) == len(fixed_hard_ids) and all(int(value) in lookup for value in fixed_hard_ids), 'Frozen hard IDs differ')
        hard = np.array([lookup[int(value)] for value in fixed_hard_ids], dtype=np.int64)
        require((valid.sum(2)[hard] > 0).all(), 'Frozen hard windows no longer have all leads')
    masks = groups(data, hard)
    overview, grouped, horizon_rows, group_horizons, direction_rows, cases = [], [], [], [], [], []
    for case, prediction in predictions.items():
        row = metrics(prediction, y, valid)
        scores, complete = window_scores(prediction, y, valid)
        row['window_p95'] = float(np.quantile(scores[complete], .95)) if complete.any() else None
        cases.append(dict(selection=selection, case=case, **row))
    for index, (name, a, b) in enumerate(COMPARISONS, 1):
        control, candidate = predictions[a], predictions[b]
        before, _ = window_scores(control, y, valid)
        after, _ = window_scores(candidate, y, valid)
        overview.append(dict(selection=selection, comparison=name, **paired(control, candidate, y, valid, count),
            fixed_hard_window_gain=float((before[hard]-after[hard]).mean()) if len(hard) else None))
        part_rows = []
        for group, mask in masks.items():
            row = dict(selection=selection, comparison=name, group=group, **paired(control, candidate, y, valid & mask, count))
            grouped.append(row); part_rows.append(row)
        for prefix in ('flow:', 'road:', 'bias_'):
            total = sum(row['global_mae_contribution'] for row in part_rows if row['group'].startswith(prefix))
            require(np.isclose(total, overview[-1]['mae_gain'], atol=1e-10, rtol=1e-10), 'Group contribution conservation failed: '+prefix)
        for h in range(12):
            horizon_rows.append(dict(selection=selection, comparison=name, step=h+1,
                **paired(control[:, h], candidate[:, h], y[:, h], valid[:, h], count)))
        if name == 'RECEIVER_MATCH_vs_POST_MESSAGE':
            for group, mask in masks.items():
                for h in range(12):
                    group_horizons.append(dict(selection=selection, group=group, step=h+1,
                        **paired(control[:, h], candidate[:, h], y[:, h], valid[:, h] & mask[:, h], count)))
            direction_rows += [dict(selection=selection, **row) for row in directions(control, candidate, y, valid, count)]
        if progress:
            print(json.dumps(dict(phase='paired_analysis', selection=selection,
                comparison=name, completed=index, total=len(COMPARISONS))), flush=True)
    identity, identity_rows = error_identity(predictions, y, valid, masks)
    before, complete = window_scores(predictions['POST_MESSAGE'], y, valid)
    after, _ = window_scores(predictions['RECEIVER_MATCH'], y, valid)
    cross_rows = []
    for road, road_mask in masks.items():
        if not road.startswith('road:'):
            continue
        for flow, flow_mask in masks.items():
            if flow.startswith('flow:'):
                cross_rows.append(dict(selection=selection, road=road, flow=flow,
                    **paired(predictions['POST_MESSAGE'], predictions['RECEIVER_MATCH'], y,
                             valid & road_mask & flow_mask, count)))
    windows = [dict(selection=selection, sample_id=int(data['sample_ids'][i]), complete=bool(complete[i]),
        original_fixed_hard=bool(i in hard), gw2_mae=None if not complete[i] else float(before[i]),
        gw3_mae=None if not complete[i] else float(after[i]), mae_gain=None if not complete[i] else float(before[i]-after[i])) for i in range(len(y))]
    for name, rows in (('cases', cases), ('comparisons', overview), ('groups', grouped), ('horizons', horizon_rows),
                      ('group_horizons', group_horizons), ('directions', direction_rows), ('windows', windows), ('error_identity', identity_rows)):
        write_csv(output/(selection+'_'+name+'.csv'), rows)
    write_csv(output/(selection+'_road_flow.csv'), cross_rows)
    return dict(validation_rows=len(y), comparisons=overview, cases=cases,
        fixed_hard_ids=data['sample_ids'][hard].tolist(), error_identity=identity,
        top_harm_groups={prefix:sorted((row for row in grouped if row['comparison'] == 'RECEIVER_MATCH_vs_POST_MESSAGE'
            and row['group'].startswith(prefix) and row['global_mae_contribution'] < 0),
            key=lambda r:r['global_mae_contribution'])[:5]
            for prefix in ('flow:', 'road:', 'bias_')})


def epoch_comparisons(rows):
    by_arm = {arm:{row['epoch']:row for row in rows if row['arm'] == arm} for arm in ARMS}
    epochs = sorted(set.intersection(*(set(value) for value in by_arm.values())))
    return [dict(epoch=epoch,
        original_mae=by_arm['ORIGINAL'][epoch]['validation_mae'],
        static_mae=by_arm['STATIC'][epoch]['validation_mae'],
        gw2_mae=by_arm['POST_MESSAGE'][epoch]['validation_mae'],
        fixed_mae=by_arm['FIXED_UNIFORM'][epoch]['validation_mae'],
        message_self_mae=by_arm['MESSAGE_SELF'][epoch]['validation_mae'],
        gw3_mae=by_arm['RECEIVER_MATCH'][epoch]['validation_mae'],
        gw3_vs_gw2_mae_gain=by_arm['POST_MESSAGE'][epoch]['validation_mae']-by_arm['RECEIVER_MATCH'][epoch]['validation_mae']) for epoch in epochs]


def build_report(analysis, histories_rows, mechanisms):
    fmt = lambda x:'不可用' if x is None else f'{x:.6f}'
    lines = ['# GW3 相对 GW2：只读损害与机制诊断', '',
        '仅读取已有预测、JSON和来源代码哈希；没有导入模型、读取原始数据或检查点、训练、新推理、测试评价。',
        '此为已用于设计的验证集探索诊断；窗口和预测点不能视为独立事故重复。',
        '收益=对照误差−候选误差；正值更好。所有真值流量、偏差及修正方向只用于事后分析。', '',
        '困难集合由best ORIGINAL最差10%完整窗口确定，best/last共用相同样本ID，不随候选或最后检查点重选。', '',
        '## 整体与同检查点依赖', '', '| 选择 | 比较 | MAE收益 | RMSE收益 | MAPE收益百分点 | 固定困难窗口收益 |', '|---|---|---:|---:|---:|---:|']
    for selection, result in analysis['evaluations'].items():
        for row in result['comparisons']:
            lines.append('| '+selection+' | '+row['comparison']+' | '+' | '.join(fmt(v) for v in (
                row['mae_gain'], row['rmse_gain'], 100*row['mape_gain'], row['fixed_hard_window_gain']))+' |')
    lines += ['', '## 相对 GW2 的损害集中在哪里', '',
        '按全网MAE贡献升序列出每类前5个负收益组；不是按组内均值挑最差的小组。完整正负组保留在CSV，分组之间不可混加贡献。', '',
        '| 选择 | 分组 | 有效点数 | 组内MAE收益 | 全网MAE贡献 | 抵消比例 |', '|---|---|---:|---:|---:|---:|']
    for selection, result in analysis['evaluations'].items():
        for rows in result['top_harm_groups'].values():
            for row in rows:
                lines.append(f"| {selection} | {row['group']} | {row['valid_count']} | {fmt(row['mae_gain'])} | {fmt(row['global_mae_contribution'])} | {fmt(row['cancellation'])} |")
    lines += ['', '## 训练曲线摘要', '', '| 组 | 最佳轮次 | 最佳MAE | 最后MAE | 最后减最佳 |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        rows = [row for row in histories_rows if row['arm'] == arm]
        best = next(row for row in rows if row['selected_best'])
        lines.append(f"| {arm} | {best['epoch']} | {fmt(best['validation_mae'])} | {fmt(rows[-1]['validation_mae'])} | {fmt(rows[-1]['validation_mae']-best['validation_mae'])} |")
    lines += ['', '| 轮次 | ORIGINAL | STATIC | GW2 | FIXED | MESSAGE_SELF | GW3 | GW3相对GW2收益 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in epoch_comparisons(histories_rows):
        lines.append('| '+str(row['epoch'])+' | '+' | '.join(fmt(row[key]) for key in (
            'original_mae', 'static_mae', 'gw2_mae', 'fixed_mae', 'message_self_mae', 'gw3_mae', 'gw3_vs_gw2_mae_gain'))+' |')
    lines += ['', '## 误差恒等式与机制可见范围', '',
        '逐点核对：完整模型MAE收益 = 关闭选择器时的MAE收益 + GW3选择器依赖 − GW2选择器依赖。',
        '这是同点绝对误差恒等式，不是训练因果分解，不能套用于RMSE，也不能证明主干被损坏。',
        '机制缓存来源链已追溯，状态见mechanism_availability.json；缺失统计标为缺失，不置零、不补推理。',
        'mechanism_phase.csv固定节点/局部窗口位置，分别汇总history和forecast调用阶段的跨样本方差。',
        '现有归档没有逐点权重极值、查询/键范数、分数或梯度，不能判定局部饱和、尺度爆炸或梯度病因。全局平均熵不能排除局部饱和。',
        'forecast_1等是原自回归卷积调用阶段，不等同于预测第1步等实际时距。',
        '动态评分关闭与QUERY_SELF替换可能分布外；STATIC_ONLY不是保留训练均值的控制。', '',
        '下一步应根据分组损害、实际时距与逐轮曲线选择一个结构假设；本工具不自动调参、选模型、发起训练或宣称新模块优于基线。', '',
        '数值制品：best_*/last_*的cases、comparisons、groups、horizons、group_horizons、road_flow、directions、windows、error_identity CSV；以及training_curves.csv、matched_epoch_curves.csv、training_mechanisms.csv、mechanism_summary.csv、mechanism_phase.csv。', '']
    if any(not row['available'] for row in mechanisms):
        lines += ['存在不可用机制统计，具体原因见mechanism_availability.json；不影响已认证预测的误差诊断。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--best-eval', type=Path)
    parser.add_argument('--last-eval', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    require(not output.exists(), 'Use a fresh diagnosis directory; existing artifacts are preserved')
    output.mkdir(parents=True)
    ledger = Inputs()
    try:
        identity, summary = ledger.json(run/'identity.json'), ledger.json(run/'summary.json')
        identity_digest(identity)
        require(identity.get('schema') == 'original_igstgnn_receiver_match_training_v1', 'Expected a GW3 run')
        require(summary.get('status') == 'GW3_TRAINING_PASS' and summary.get('same_completed_training_budget') is True,
                'All groups must have a completed common budget')
        require(summary.get('engineering_only') is False, 'Use full validation, not miniature engineering runs')
        require(all(summary['arms'][arm]['completed_epoch'] == summary['common_completed_epochs'] for arm in ARMS), 'Completed budgets differ')
        analysis = dict(schema=SCHEMA, status='GW3_READ_ONLY_DIAGNOSIS_PASS',
            created_utc=datetime.now(timezone.utc).isoformat(), evaluations={}, new_inference_calls=0,
            optimizer_updates=0, test_evaluated=False, checkpoint_files_read=0,
            analysis_source_sha256=ledger.record(__file__), numpy_version=str(np.__version__),
            certification='cached_archive_and_saved_history_consistency_not_new_checkpoint_replay')
        mechanism_summary, mechanism_phase, availability, reference = [], [], [], None
        frozen_hard_ids, saved_axes = None, None
        for selection in ('best', 'last'):
            print(json.dumps(dict(phase='read_certified_archives', selection=selection)), flush=True)
            directory, meta = find_evaluation(run, identity, summary, selection, ledger, getattr(args, selection+'_eval'))
            require(meta['reference'] == summary['reference'], 'Evaluation/summary source reference differs')
            reference = meta['reference']
            source_identities(reference, identity, ledger)
            data, predictions, old_directory = load_pair(directory, meta, ledger, selection)
            axes = {key:hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()
                +str(data[key].shape).encode()+str(data[key].dtype).encode()).hexdigest() for key in AXES}
            if saved_axes is None:
                saved_axes = axes
            require(saved_axes == axes, 'best/last targets, masks or axes differ')
            for arm in ARMS:
                measured = metrics(predictions[arm], data['y'], data['valid'])
                expected = summary['arms'][arm]['best_metrics' if selection == 'best' else 'last_metrics']
                for metric, key in (('mae', 'mae'), ('rmse', 'rmse'), ('mape', 'mape_fraction')):
                    require(np.isclose(measured[metric], expected[key], atol=1e-5, rtol=1e-6), 'Cached metrics differ from summary: '+arm)
            analysis['evaluations'][selection] = analyze_pair(selection, data, predictions, output,
                progress=True, fixed_hard_ids=frozen_hard_ids)
            frozen_hard_ids = analysis['evaluations'][selection]['fixed_hard_ids']
            allowed = (run, Path(reference['source_run']))
            for case in ('POST_MESSAGE', 'MESSAGE_SELF', 'RECEIVER_MATCH', 'RECEIVER_MATCH_UNIFORM', 'RECEIVER_MATCH_QUERY_SELF'):
                try:
                    stats, chain = resolve_mechanisms(directory, case, ledger, allowed)
                    a, b = flatten_mechanisms(stats, selection=selection, case=case)
                    mechanism_summary += a; mechanism_phase += b
                    availability.append(dict(selection=selection, case=case, available=True, source_chain=chain))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    availability.append(dict(selection=selection, case=case, available=False, reason=str(error)))
            del data, predictions
            gc.collect()
        history, training_mechanisms = histories(run, identity, summary, reference, ledger)
        write_csv(output/'training_curves.csv', history)
        write_csv(output/'matched_epoch_curves.csv', epoch_comparisons(history))
        write_csv(output/'training_mechanisms.csv', training_mechanisms)
        write_csv(output/'mechanism_summary.csv', mechanism_summary)
        write_csv(output/'mechanism_phase.csv', mechanism_phase)
        write_json(output/'mechanism_availability.json', availability)
        report_text = build_report(analysis, history, availability)
        print(json.dumps(dict(phase='verify_input_preservation', source_files=len(ledger.files))), flush=True)
        preserved = ledger.preserved()
        write_json(output/'preservation.json', preserved)
        write_json(output/'analysis.json', analysis)
        (output/'report.md').write_text(report_text, encoding='utf-8')
        write_json(output/'summary.json', dict(schema=SCHEMA, status=analysis['status'],
            validation_rows={key:value['validation_rows'] for key, value in analysis['evaluations'].items()},
            new_inference_calls=0, optimizer_updates=0, test_evaluated=False,
            source_artifacts_preserved=True, mechanism_missing_cases=sum(not row['available'] for row in availability),
            output_dir=str(output)))
        print(json.dumps(dict(status=analysis['status'], output_dir=str(output), new_inference_calls=0)), flush=True)
        return analysis
    except BaseException as error:
        write_json(output/'failure.json', dict(status='GW3_READ_ONLY_DIAGNOSIS_FAILED', error_type=type(error).__name__,
            error=str(error), new_inference_calls=0, optimizer_updates=0, test_evaluated=False))
        raise


if __name__ == '__main__':
    main()
