"""Read-only authentication for the returned failed GW11 LAST experiment.

Checkpoints are hashed as opaque files. This module does not import Torch,
deserialize a checkpoint, execute inference, select a winner, or fit anything.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as source_auth
from experiments.conditional_residual_gw10.screen import metrics
from experiments.state_gate_gw11 import protocol
from experiments.state_gate_gw11.state import SCHEMA, identity_digest, epoch_order, epoch_seed, require


ARMS = protocol.ARMS
CASES = ('FROZEN_SUM', 'ORIGIN', 'OWN4', *ARMS)
AXIS_KEYS = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')
ARCHIVE_SCHEMA = 'gw11_state_gate_prediction_archive_v1'
PRODUCER_SCHEMA = 'gw11_state_gate_prediction_producer_v1'
PRODUCTION_SHAPE = (837, 12, 496)
SOURCE_IDENTITY_SHA256 = 'dde9b123f6eae32e60760ff360dcd2caf9da252706b5b1a0e6031f77ea6921d7'
EXPECTED_EVALUATIONS = {
    'best': dict(metadata_sha256='965c912f5e3f35a734a9e2a41bf95569816e6836f1c1b81d4299be002e40cea2',
        archive_sha256='17795f0776a9b408c3b6a43073e6a0a044500e05e94de795f74579d26c53b5d1'),
    'last': dict(metadata_sha256='11917307d5db6938ca9ec5602d2cbd6cbb3b2a3028e9f6f618996b674e84cb03',
        archive_sha256='a4ae77f42683fc92c5f1791889935c5423cda9e7f0e4f1bd3d86b31ed6dd56ec')}
EXPECTED_CHECKS = {
    **{'BEST_relative_gain_vs_'+arm+'_at_least_0_005': False for arm in ARMS[:-1]},
    'BEST_no_worse_than_frozen_GW10_OWN4': False,
    'LAST_own_degradation_at_most_0_01': False,
    **{'BEST_'+group+'_no_worse_than_SUM_CONTINUE': True for group in protocol.GROUPS}}


def sha256(path):
    return source_auth.sha256(path)


def _json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _capture(path, digests, expected=None):
    path = Path(path).resolve()
    require(path.is_file(), 'Missing immutable GW11 file: '+str(path))
    actual = sha256(path)
    require(expected is None or actual == expected, 'GW11 file digest differs: '+str(path))
    require(str(path) not in digests or digests[str(path)] == actual,
        'GW11 file changed while authenticating: '+str(path))
    digests[str(path)] = actual
    return path


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(character in '0123456789abcdef' for character in value)


def _child(directory, name):
    require(isinstance(name, str) and name not in ('', '.', '..')
        and Path(name).name == name, 'Expected a direct checkpoint basename')
    path = (Path(directory)/name).resolve()
    require(path.parent == Path(directory).resolve(), 'Checkpoint escapes its arm directory')
    return path


def _digest_maps(value, digests):
    """Capture already identity-bound file maps without loading their contents."""
    if isinstance(value, list):
        for child in value:
            _digest_maps(child, digests)
    elif isinstance(value, dict):
        for name, child in value.items():
            if name in ('file_digests', 'cache_file_digests'):
                require(isinstance(child, dict), 'Invalid nested file digest map')
                for filename, expected in child.items():
                    require(isinstance(filename, str) and Path(filename).is_absolute()
                        and _is_digest(expected),
                        'Nested digests require absolute files and SHA256')
                    _capture(filename, digests, expected)
            else:
                _digest_maps(child, digests)


def _scalar(archive, name):
    value = np.asarray(archive.get(name))
    require(value.ndim == 0 and value.dtype.kind in 'US', 'Invalid archive scalar: '+name)
    return str(value.item())


def validate_archive(archive):
    require(_scalar(archive, 'schema') == ARCHIVE_SCHEMA, 'Expected the GW11 prediction archive schema')
    require(_scalar(archive, 'axis_order') == 'sample,horizon,node', 'Explicit GW11 axes required')
    require(set(archive) == {*AXIS_KEYS, 'schema', 'axis_order', *('pred__'+case for case in CASES)},
        'Exactly the seven GW11 cases and shared axes are required')
    y, valid = np.asarray(archive['y']), np.asarray(archive['valid'])
    require(y.shape == PRODUCTION_SHAPE and y.dtype == np.float32 and np.isfinite(y).all(),
        'Expected complete float32 GW11 targets with shape 837x12x496')
    require(valid.dtype == np.bool_ and valid.shape == y.shape and bool(valid.any()),
        'Expected the shared nonempty Boolean validity mask')
    null = float(y.min()) if y.min() < 1 else 0.
    require(np.array_equal(valid, y != null) and bool((y[valid] > 0).all()),
        'GW11 native global-null mask or positive MAPE targets differ')
    for name, count in (('sample_ids', y.shape[0]), ('node_ids', y.shape[2])):
        ids = np.asarray(archive[name])
        require(ids.shape == (count,) and ids.dtype.kind in 'iu' and np.unique(ids).size == count,
            'Invalid GW11 unique axis: '+name)
    roads, support = np.asarray(archive['road_labels']), np.asarray(archive['direct_support'])
    require(roads.shape == (y.shape[2],) and roads.dtype.kind in 'US', 'Invalid GW11 road labels')
    require(support.shape == (y.shape[0], y.shape[2]) and support.dtype == np.bool_,
        'Invalid GW11 direct-support axis')
    for case in CASES:
        prediction = np.asarray(archive['pred__'+case])
        expected = np.float64 if case in ('ORIGIN', 'OWN4') else np.float32
        require(prediction.shape == y.shape and prediction.dtype == expected
            and bool(np.isfinite(prediction).all()), 'Invalid GW11 prediction: '+case)
    return y, valid


def _record(prediction, truth, valid):
    row = metrics(prediction, truth, valid)
    selections = dict(high_value_ge314=truth >= 314,
        long_horizon_7_12=np.arange(12)[None, :, None] >= 6,
        flow_50_200=(truth >= 50) & (truth < 200))
    row['groups'] = {}
    for name in protocol.GROUPS:
        selected = valid & np.broadcast_to(selections[name], valid.shape)
        row['groups'][name] = (metrics(prediction, truth, selected) if selected.any()
            else dict(valid_count=0, mae=None, signed_bias=None))
    counts = valid.sum(axis=(1, 2)); complete = counts > 0
    scores = np.divide((np.abs(prediction.astype(np.float64)-truth)*valid).sum(axis=(1, 2)),
        counts, out=np.zeros(len(counts), dtype=np.float64), where=complete)
    row['window_p95'] = float(np.quantile(scores[complete], .95)) if complete.any() else None
    return row


def _same_record(actual, expected, context):
    require(isinstance(expected, dict) and set(actual) == set(expected), 'GW11 metric fields differ: '+context)
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, dict):
            _same_record(value, other, context+'/'+key)
        elif value is None:
            require(other is None, 'GW11 unavailable metric differs: '+context+'/'+key)
        elif key == 'valid_count':
            require(type(other) is int and other == value, 'GW11 count differs: '+context+'/'+key)
        else:
            require(isinstance(other, (int, float)) and not isinstance(other, bool)
                and np.isfinite(other) and bool(np.isclose(value, other, rtol=1e-10, atol=1e-10)),
                'GW11 metric replay differs: '+context+'/'+key)


def _same_array(left, right, name):
    require(left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right),
        'GW11 BEST/LAST axes or frozen prediction differ: '+name)


def _source_paths(identity):
    source = identity.get('source_certificate', {})
    gate = identity.get('mask_gate', {})
    cache = gate.get('bindings', {}).get('cache_file_digests', {})
    trains = [Path(filename).resolve() for filename in cache if Path(filename).name == 'train.npz']
    require(len(trains) == 1, 'Exactly one bound training cache is required')
    values = dict(probe_dir=source.get('probe_directory'), cache_dir=str(trains[0].parent),
        gw10_dir=identity.get('gw10_reference', {}).get('directory'), audit_dir=gate.get('directory'))
    require(all(isinstance(value, str) and Path(value).is_absolute() for value in values.values()),
        'GW11 source directories must be explicit absolute paths')
    return {key: str(Path(value).resolve()) for key, value in values.items()}


def load_run(run_dir, repo=None):
    """Return LAST arrays, immutable identity, and an opaque-file certificate."""
    directory = Path(run_dir).resolve()
    repo = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2]
    digests, protected = {}, {str(directory)}
    identity = _json(_capture(directory/'identity.json', digests))
    require(identity_digest(identity) == identity.get('identity_sha256') == SOURCE_IDENTITY_SHA256,
        'Only the returned completed GW11 development run is accepted')
    require(identity.get('schema') == SCHEMA and identity.get('arms') == list(ARMS)
        and identity.get('candidate') == 'STATE_GATE' and identity.get('new_epochs') == 3
        and identity.get('train_batches') == 82 and identity.get('train_count') == 3909
        and identity.get('full_horizon') == 12 and identity.get('seed') == 2025
        and identity.get('optimizer_reset_at_start') is True
        and identity.get('best_selection_includes_stage_zero') is True
        and identity.get('test_evaluated') is False and identity.get('extension_allowed') is False,
        'GW11 identity protocol differs')
    execution = identity.get('execution_source', {})
    require(isinstance(execution, dict) and bool(execution), 'Missing bound GW11 execution source')
    for name, expected in execution.items():
        path = (repo/name).resolve()
        require(isinstance(name, str) and not Path(name).is_absolute() and repo in path.parents,
            'GW11 execution source escapes the repository')
        _capture(path, digests, expected)
    paths = _source_paths(identity)
    protected.update(paths.values())
    source_auth._source_tree(identity['source_certificate'], digests, protected)
    _digest_maps(identity, digests)
    protected.update(identity['source_certificate'].get('protected_directories', []))
    summary = _json(_capture(directory/'summary.json', digests))
    require(summary.get('schema') == SCHEMA and summary.get('status') == 'GW11_TRAINING_COMPLETE'
        and summary.get('identity_sha256') == SOURCE_IDENTITY_SHA256
        and summary.get('fixed_new_epochs') == 3 and summary.get('registered_arms') == list(ARMS)
        and summary.get('completed_epochs') == {arm: 3 for arm in ARMS}
        and summary.get('updates') == {arm: 246 for arm in ARMS}
        and all(type(value) is int for name in ('completed_epochs', 'updates')
            for value in summary[name].values())
        and set(summary.get('best', {})) == set(ARMS)
        and summary.get('best_selection_includes_stage_zero') is True
        and summary.get('optimizer_reset_at_start') is True
        and summary.get('optimizer_reset_on_resume') is False
        and summary.get('test_evaluated') is False and summary.get('extension_allowed') is False,
        'GW11 completed summary or fixed budget differs')
    marker = _json(_capture(directory/'training_started.json', digests))
    require(marker == dict(schema='gw11_training_started_marker_v1', training_schema=SCHEMA,
        identity_sha256=SOURCE_IDENTITY_SHA256, registered_arms=list(ARMS), fixed_new_epochs=3,
        all_stage_zero_checkpoints_committed=True, written_before_first_optimizer_update=True),
        'GW11 original training-start marker differs')
    decision = _json(_capture(directory/'screening.json', digests))
    require(decision.get('run_identity_sha256') == SOURCE_IDENTITY_SHA256
        and decision.get('candidate_for_confirmation') is False
        and decision.get('eligible_for_independent_confirmation') is False
        and decision.get('checks') == EXPECTED_CHECKS and decision.get('test_evaluated') is False,
        'GW11 original failed screening differs')
    pointer = _json(_capture(directory/'result_pointer.json', digests))
    selected = Path(pointer.get('directory', '')).resolve()
    require(selected.parent == directory and selected.name.startswith('evaluation_'),
        'GW11 result pointer must bind one direct immutable evaluation child')
    require(_is_digest(pointer.get('report_sha256')) and _is_digest(pointer.get('screening_sha256')),
        'GW11 result pointer requires both original output digests')
    _capture(selected/'report.md', digests, pointer.get('report_sha256'))
    _capture(selected/'screening.json', digests, pointer.get('screening_sha256'))
    require(_json(selected/'screening.json') == decision, 'GW11 nested screening differs')
    require((directory/'report.md').read_bytes() == (selected/'report.md').read_bytes(),
        'GW11 root report differs from the bound evaluation report')
    _capture(directory/'report.md', digests)
    require(set(decision.get('evaluations', {})) == {'best', 'last'}, 'GW11 evaluation roles differ')
    archives, metadata, records = {}, {}, {}
    for selection in ('best', 'last'):
        child = selected/selection
        bound = decision['evaluations'][selection]
        pins = EXPECTED_EVALUATIONS[selection]
        require(Path(bound.get('directory', '')).resolve() == child
            and bound.get('metadata_sha256') == pins['metadata_sha256']
            and bound.get('archive_sha256') == pins['archive_sha256'],
            'GW11 screening must reference the original returned evaluation: '+selection)
        meta = _json(_capture(child/'metadata.json', digests, pins['metadata_sha256']))
        require(meta.get('schema') == PRODUCER_SCHEMA and meta.get('checkpoint_selection') == selection
            and meta.get('run_identity_sha256') == SOURCE_IDENTITY_SHA256
            and meta.get('archive_sha256') == pins['archive_sha256']
            and meta.get('common_completed_new_epochs') == 3
            and meta.get('optimizer_updates_this_evaluation') == 0 and meta.get('test_evaluated') is False,
            'GW11 producer metadata differs: '+selection)
        roles = meta.get('checkpoint_roles', {})
        require(set(roles) == set(ARMS), 'Exactly four GW11 checkpoint roles are required')
        for arm, role in roles.items():
            require(role.get('selection') == selection and type(role.get('epoch')) is int
                and role.get('epoch') == (0 if selection == 'best' else 3)
                and _is_digest(role.get('model_tensor_sha256'))
                and _is_digest(role.get('checkpoint_sha256')), 'GW11 checkpoint epoch/tensor role differs: '+arm)
            if selection == 'best':
                require({key: value for key, value in role.items() if key != 'selection'} == summary.get('best', {}).get(arm),
                    'GW11 BEST summary role differs: '+arm)
            else:
                require(role.get('checkpoint_file') == 'last.pt', 'GW11 LAST must use last.pt')
            _capture(_child(directory/arm, role.get('checkpoint_file')), digests, role.get('checkpoint_sha256'))
        archive_path = _capture(child/'paired_predictions.npz', digests, pins['archive_sha256'])
        with np.load(archive_path, allow_pickle=False) as payload:
            archive = {key: payload[key].copy() for key in payload.files}
        y, valid = validate_archive(archive)
        require(np.array_equal(archive['sample_ids'], identity['data']['splits']['val']['selected_indices']),
            'GW11 validation sample order differs from its data identity')
        rows = {case: _record(archive['pred__'+case], y, valid) for case in CASES}
        require(set(meta.get('records', {})) == set(ARMS)
            and set(meta.get('historical_records', {})) == {'FROZEN_SUM', 'ORIGIN', 'OWN4'},
            'GW11 producer metric cases differ')
        for case in CASES:
            _same_record(rows[case], {**meta['historical_records'], **meta['records']}[case], selection+'/'+case)
        report_metrics = _json(_capture(child/'metrics.json', digests))
        require(set(report_metrics) == set(CASES), 'GW11 metrics JSON cases differ')
        for case in CASES:
            _same_record(rows[case], report_metrics[case], 'metrics/'+selection+'/'+case)
        archives[selection], metadata[selection], records[selection] = archive, meta, rows
    for name in (*AXIS_KEYS, 'pred__FROZEN_SUM', 'pred__ORIGIN', 'pred__OWN4'):
        _same_array(archives['best'][name], archives['last'][name], name)
    for arm in ARMS:
        _same_array(archives['best']['pred__'+arm], archives['best']['pred__FROZEN_SUM'], 'BEST/'+arm)
        history = _json(_capture(directory/arm/'history.json', digests))
        require(isinstance(history, list) and len(history) == 3
            and all(isinstance(row, dict) and type(row.get('epoch')) is int for row in history)
            and [row.get('epoch') for row in history] == [1, 2, 3], 'GW11 arm history epochs differ: '+arm)
        for row in history:
            order = epoch_order(identity['seed'], row['epoch'], identity['train_count'])
            require(type(row.get('global_updates')) is int and row.get('global_updates') == 82*row['epoch']
                and row.get('train_order_sha256') == hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
                and row.get('dropout_epoch_seed') == epoch_seed(identity['seed'], row['epoch'])
                and row.get('mask_difference_points') == 0,
                'GW11 history shared order, updates, RNG or masks differ: '+arm)
            require(row.get('validation', {}).get('mae', -1) >= records['best'][arm]['mae'],
                'GW11 history contradicts the selected stage-zero BEST: '+arm)
        _same_record(records['last'][arm], history[-1]['validation'], 'history/LAST/'+arm)
        require(metadata['last']['checkpoint_roles'][arm]['mae'] == history[-1]['validation']['mae'],
            'GW11 LAST checkpoint score differs from history: '+arm)
    recomputed = protocol.screen({arm: records['best'][arm] for arm in ARMS},
        {arm: records['last'][arm] for arm in ARMS}, {'OWN4': records['best']['OWN4']})
    require({key: value for key, value in decision.items() if key not in ('run_identity_sha256', 'evaluations')}
        == recomputed, 'GW11 archive-derived original failed screening differs')
    certificate = dict(schema='gw11_last_ablation_authentication_v1', directory=str(directory),
        source_identity_sha256=SOURCE_IDENTITY_SHA256, file_digests=digests,
        protected_directories=sorted(protected), source_paths=paths, summary=summary,
        best_metadata=metadata['best'], last_metadata=metadata['last'], screening=decision,
        evaluation_directory=str(selected), data_identity=identity['data'],
        aggregate_records=records, training_updates_this_authentication=0,
        checkpoint_deserializations=0, neural_inference_calls=0,
        source_artifacts_written=False, test_evaluated=False)
    assert_unchanged(certificate)
    return archives['last'], identity, certificate


def assert_unchanged(certificate):
    for filename, expected in certificate['file_digests'].items():
        require(Path(filename).is_file() and sha256(filename) == expected,
            'Authenticated GW11 source changed: '+filename)


def require_output_outside_sources(output, run_dir, certificate, data_dir=None):
    candidate = Path(output).resolve()
    require(not candidate.exists(), 'Use a fresh GW11 ablation output directory')
    protected = {Path(run_dir).resolve(), *(Path(value).resolve()
        for value in certificate.get('protected_directories', [])),
        *(Path(value).resolve() for value in certificate.get('source_paths', {}).values())}
    if data_dir is not None:
        protected.add(Path(data_dir).resolve())
    for source in protected:
        require(candidate != source and source not in candidate.parents and candidate not in source.parents,
            'GW11 ablation output overlaps an immutable source/data directory: '+str(source))
    return candidate
