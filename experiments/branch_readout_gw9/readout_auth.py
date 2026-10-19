"""Authenticate the returned, failed GW9 probe using bytes and NumPy only.

This reader is deliberately specific to one completed development experiment.
It hashes checkpoints as opaque bytes; it never deserializes a checkpoint,
imports a model, runs inference, or changes the failed training gate.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'original_igstgnn_branch_readout_gw9_probe_v1'
ARCHIVE_SCHEMA = 'original_igstgnn_branch_readout_gw9_probe_archive_v1'
PRODUCER_SCHEMA = 'gw9_frozen_branch_readout_probe_producer_v1'
ARMS = ('SUM_CONTROL', 'BRANCH_READOUT')
CASES = ('FROZEN_R', *ARMS)
AXIS_KEYS = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')
PRODUCTION_SHAPE = (837, 12, 496)
SOURCE_IDENTITY_SHA256 = '5e20efbdcca907afda994889cdc460276fdd560e235047a11c20743494d36757'
LAST_CHECKPOINT_SHA256 = '359c1384e70a5ec810dc2aa378e6fd2f6be2f9105ead36c2cdddd4235f3a4e5c'
EXPECTED_EVALUATIONS = {
    'best': dict(metadata_sha256='ae0b45df86f9f202b9f28bd0c78a6decbf38ff30eaa14147f2b0291822350ef2',
        archive_sha256='b66cedc58d29be5e4df3f060c2c0db7cd4366470d6808b40c2dec38e02b52eec'),
    'last': dict(metadata_sha256='2e7b62e7952655a2a8cb77a0b6c93d6996f719c64a75ad7ba24290847ce29f50',
        archive_sha256='b66cedc58d29be5e4df3f060c2c0db7cd4366470d6808b40c2dec38e02b52eec')}
EXPECTED_CHECKS = {
    'BEST_relative_gain_vs_SUM_at_least_0_005': False,
    'BEST_relative_gain_vs_frozen_R_at_least_0_005': True,
    'LAST_own_degradation_at_most_0_01': True,
    'BEST_high_value_ge314_no_worse_than_frozen_R': True,
    'BEST_long_horizon_7_12_no_worse_than_frozen_R': True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _identity_digest(identity):
    value = {key: item for key, item in identity.items() if key != 'identity_sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _scalar(archive, name):
    value = np.asarray(archive.get(name))
    require(value.ndim == 0 and value.dtype.kind in 'US', 'Invalid scalar '+name)
    return str(value.item())


def validate_archive(archive):
    """Validate all axes and exact cases; production shape is checked by the reader."""
    require(_scalar(archive, 'schema') == ARCHIVE_SCHEMA, 'Not a GW9 probe archive')
    require(_scalar(archive, 'axis_order') == 'sample,horizon,node', 'Explicit prediction axes required')
    required = {*AXIS_KEYS, 'schema', 'axis_order', 'registered_arms', *('pred__'+case for case in CASES)}
    require(set(archive) == required, 'Exactly the GW9 probe keys and cases are required')
    require(np.array_equal(np.asarray(archive['registered_arms']), np.asarray(ARMS)), 'Probe arms differ')
    y, valid = np.asarray(archive['y']), np.asarray(archive['valid'])
    require(y.ndim == 3 and y.shape[1] == 12 and min(y.shape) > 0 and y.dtype.kind == 'f',
        'Expected floating [sample,12,node] targets')
    require(valid.shape == y.shape and valid.dtype == np.bool_ and valid.any(), 'Shared Boolean mask required')
    require(np.isfinite(y).all() and (y[valid] > 0).all(), 'Targets must be finite and valid MAPE targets positive')
    null = float(y.min()) if y.min() < 1 else 0.
    require(np.array_equal(valid, y != null), 'Mask differs from the native global-null rule')
    for name, count in (('sample_ids', y.shape[0]), ('node_ids', y.shape[2])):
        ids = np.asarray(archive[name])
        require(ids.shape == (count,) and ids.dtype.kind in 'iu' and np.unique(ids).size == count,
            'Invalid unique '+name)
    roads, support = np.asarray(archive['road_labels']), np.asarray(archive['direct_support'])
    require(roads.shape == (y.shape[2],) and roads.dtype.kind in 'US', 'Invalid road axis')
    require(support.shape == (y.shape[0], y.shape[2]) and support.dtype == np.bool_, 'Invalid direct-support axis')
    predictions = {case: np.asarray(archive['pred__'+case]) for case in CASES}
    for case, prediction in predictions.items():
        require(prediction.shape == y.shape and prediction.dtype.kind == 'f'
            and np.isfinite(prediction).all(), 'Invalid prediction axes or values: '+case)
    return y, valid, predictions


def _same_array(actual, expected, name):
    require(actual.dtype == expected.dtype and actual.shape == expected.shape
        and np.array_equal(actual, expected), 'Array differs from authenticated source: '+name)


def _capture(path, digests, expected=None):
    path = Path(path).resolve()
    require(path.is_file(), 'Missing immutable source file: '+str(path))
    actual = sha256(path)
    require(expected is None or actual == expected, 'Source file digest differs: '+str(path))
    prior = digests.setdefault(str(path), actual)
    require(prior == actual, 'Source file changed while reading: '+str(path))
    return path


def _child(directory, name):
    require(isinstance(name, str) and Path(name).name == name and name not in ('', '.', '..'),
        'Expected a direct child filename')
    path = (directory/name).resolve()
    require(path.parent == directory, 'Source child escapes its directory')
    return path


def _source_tree(value, digests, protected):
    """Bind locatable files in nested, already-pinned source certificates."""
    if isinstance(value, list):
        for item in value:
            _source_tree(item, digests, protected)
        return
    if not isinstance(value, dict):
        return
    if value.get('source_run'):
        run = Path(value['source_run']).resolve()
        protected.add(str(run))
        for key, filename in (('source_identity_file_sha256', 'identity.json'),
                ('source_budget_file_sha256', 'budget.json'), ('source_summary_file_sha256', 'summary.json')):
            if key in value:
                _capture(run/filename, digests, value[key])
        if 'source_identity_file_sha256' in value:
            identity = _read_json(run/'identity.json')
            require(_identity_digest(identity) == identity.get('identity_sha256')
                == value.get('source_identity_sha256'), 'Nested source identity differs')
    if value.get('directory'):
        directory = Path(value['directory']).resolve()
        protected.add(str(directory))
        for key, filename in (('metadata_sha256', 'metadata.json'), ('archive_sha256', 'paired_predictions.npz'),
                ('summary_sha256', 'summary.json'), ('input_manifest_sha256', 'input_manifest.json')):
            if key in value:
                _capture(directory/filename, digests, value[key])
    # The oldest GW1 reference uses explicit archive/metadata filenames rather
    # than the later directory/archive_sha256 certificate representation.
    if value.get('path') and value.get('sha256'):
        archive = Path(value['path']).resolve()
        protected.add(str(archive.parent))
        _capture(archive, digests, value['sha256'])
    if value.get('metadata_path') and value.get('metadata_sha256'):
        metadata = Path(value['metadata_path']).resolve()
        protected.add(str(metadata.parent))
        _capture(metadata, digests, value['metadata_sha256'])
    if value.get('checkpoint_file') and value.get('checkpoint_sha256'):
        checkpoint = Path(value['checkpoint_file'])
        # Historical source initialization certificates use absolute filenames.
        # Relative BEST names are checked against their own run by the caller.
        if checkpoint.is_absolute():
            protected.add(str(checkpoint.resolve().parent))
            _capture(checkpoint, digests, value['checkpoint_sha256'])
    if value.get('source_checkpoint_file') and value.get('source_checkpoint_sha256'):
        checkpoint = Path(value['source_checkpoint_file']).resolve()
        protected.add(str(checkpoint.parent))
        _capture(checkpoint, digests, value['source_checkpoint_sha256'])
    for item in value.values():
        if isinstance(item, (dict, list)):
            _source_tree(item, digests, protected)


def _metrics(archive):
    y, valid, predictions = validate_archive(archive)
    result = {}
    for case, prediction in predictions.items():
        error = prediction[valid].astype(np.float64)-y[valid].astype(np.float64)
        groups = {}
        for name, selection in (('high_value_ge314', valid & (y >= 314)),
                ('long_horizon_7_12', valid & (np.arange(12)[None, :, None] >= 6))):
            count = int(selection.sum())
            groups[name] = dict(valid_count=count, mae=float(np.abs(
                prediction[selection].astype(np.float64)-y[selection].astype(np.float64)).mean()) if count else None)
        result[case] = dict(mae_pooled=float(np.abs(error).mean()),
            rmse_pooled=float(np.sqrt(np.square(error).mean())),
            mape_pooled=float((np.abs(error)/y[valid].astype(np.float64)).mean()), groups=groups)
    return result


def _screen_checks(best, last):
    b, s, r = (best[case]['mae_pooled'] for case in ('BRANCH_READOUT', 'SUM_CONTROL', 'FROZEN_R'))
    require(min(b, s, r) > 0, 'Positive MAE required for relative screening')
    result = {'BEST_relative_gain_vs_SUM_at_least_0_005': (s-b)/s >= .005,
        'BEST_relative_gain_vs_frozen_R_at_least_0_005': (r-b)/r >= .005,
        'LAST_own_degradation_at_most_0_01': (last['BRANCH_READOUT']['mae_pooled']-b)/b <= .01}
    for name in ('high_value_ge314', 'long_horizon_7_12'):
        a, control = best['BRANCH_READOUT']['groups'][name]['mae'], best['FROZEN_R']['groups'][name]['mae']
        result['BEST_'+name+'_no_worse_than_frozen_R'] = a is not None and control is not None and a <= control
    return {key: bool(value) for key, value in result.items()}


def load_authenticated_run(probe_dir, repo=None):
    """Return BEST, LAST, and a JSON-safe certificate for the pinned failed probe."""
    directory = Path(probe_dir).resolve()
    repo = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[2]
    digests, protected = {}, {str(directory)}
    identity_path = _capture(directory/'identity.json', digests)
    identity = _read_json(identity_path)
    require(_identity_digest(identity) == identity.get('identity_sha256') == SOURCE_IDENTITY_SHA256,
        'Only the returned GW9 replayfix development probe is accepted')
    require(identity.get('schema') == SCHEMA and identity.get('arms') == list(ARMS)
        and identity.get('fixed_screen_epochs') == 3 and identity.get('streaming_features') is True
        and identity.get('best_selection_includes_stage_zero') is True
        and identity.get('optimizer_reset_at_start') is True
        and identity.get('frozen_R_is_equal_budget_control') is False and identity.get('test_evaluated') is False,
        'Probe identity protocol differs')
    initial = identity.get('initializations', {})
    require(set(initial) == set(ARMS) and len({row.get('head_tensor_sha256') for row in initial.values()}) == 1
        and all(row.get('registered_parameters') == row.get('trainable_parameters') == 8256
            for row in initial.values()), 'Probe initialization capacity or tensors differ')
    execution = identity.get('execution_source', {})
    require(bool(execution), 'Missing bound execution source')
    for name, expected in execution.items():
        path = (repo/name).resolve()
        require(repo in path.parents, 'Execution source escapes repository')
        _capture(path, digests, expected)
    budget = _read_json(_capture(directory/'budget.json', digests))
    require(budget == dict(target_completed_epochs=3, extension_allowed=False, backbone_updates=0,
        registered_heads=list(ARMS), test_evaluated=False), 'Probe budget differs')
    _capture(directory/'last.pt', digests, LAST_CHECKPOINT_SHA256)
    summary = _read_json(_capture(directory/'summary.json', digests))
    history = _read_json(_capture(directory/'history.json', digests))
    expected_updates = {arm:246 for arm in ARMS}
    require(summary.get('schema') == SCHEMA and summary.get('status') == 'GW9_PROBE_TRAINING_COMPLETE'
        and summary.get('completed_epoch') == summary.get('target_completed_epochs') == 3
        and summary.get('head_updates') == expected_updates and summary.get('backbone_updates') == 0
        and summary.get('source_reference') == identity.get('source_reference')
        and summary.get('same_budget_comparison') == list(ARMS)
        and summary.get('best_selection_includes_stage_zero') is True
        and summary.get('extension_allowed') is False and summary.get('joint_training_started') is False
        and summary.get('frozen_R_is_equal_budget_control') is False and summary.get('test_evaluated') is False,
        'Completed probe summary differs')
    require(isinstance(history, list) and len(history) == 3
        and [row.get('epoch') for row in history] == [1,2,3]
        and all(row.get('head_updates') == {arm:82*row['epoch'] for arm in ARMS} for row in history),
        'Completed paired history or update budget differs')
    source = identity['source_reference']
    require(identity.get('backbone_tensor_sha256') == source['source_initialization']['model_tensor_sha256'],
        'Frozen backbone identity differs')
    _source_tree(source, digests, protected)
    source_identity = _read_json(Path(source['source_run'])/'identity.json')
    require(source_identity.get('data') == identity['data'], 'Probe/source data identity differs')
    pointer = _read_json(_capture(directory/'screening.json', digests))
    require(pointer.get('status') == 'GW9_PROBE_SCREENING_COMPLETE'
        and pointer.get('run_identity_sha256') == SOURCE_IDENTITY_SHA256
        and pointer.get('last_checkpoint_sha256') == LAST_CHECKPOINT_SHA256
        and pointer.get('evaluations') == EXPECTED_EVALUATIONS and pointer.get('checks') == EXPECTED_CHECKS
        and pointer.get('eligible_for_joint_screen') is False and pointer.get('candidate_for_confirmation') is False
        and pointer.get('joint_training_started') is False and pointer.get('frozen_R_is_equal_budget_control') is False
        and pointer.get('screening_failure_does_not_authorize_budget_extension') is True
        and pointer.get('interpretation') == 'single_seed_development_probe_not_independent_confirmation'
        and pointer.get('test_evaluated') is False, 'Pinned failed screening result differs')
    require(summary.get('screening') == pointer, 'Summary screening pointer differs')
    output = _child(directory, pointer.get('evaluation_directory'))
    require(output.name.startswith('evaluation_'), 'Expected immutable probe evaluation directory')
    nested_pointer = _read_json(_capture(output/'screening.json', digests))
    require(nested_pointer == pointer, 'Nested screening result differs')
    archives, roles = {}, {}
    baseline_info = source['archives']['best']
    baseline_path = Path(baseline_info['directory']).resolve()/'paired_predictions.npz'
    _capture(baseline_path, digests, baseline_info['archive_sha256'])
    with np.load(baseline_path, allow_pickle=False) as payload:
        baseline = {key: payload[key].copy() for key in AXIS_KEYS}
        baseline['pred__FROZEN_R'] = payload['pred__R_CONTINUE'].copy()
    for selection in ('best', 'last'):
        child = output/selection
        expected = EXPECTED_EVALUATIONS[selection]
        metadata_path = _capture(child/'metadata.json', digests, expected['metadata_sha256'])
        archive_path = _capture(child/'paired_predictions.npz', digests, expected['archive_sha256'])
        metadata = _read_json(metadata_path)
        require(metadata.get('schema') == PRODUCER_SCHEMA and metadata.get('status') == 'GW9_PROBE_PREDICTION_PASS'
            and metadata.get('checkpoint_selection') == selection
            and metadata.get('run_identity_sha256') == SOURCE_IDENTITY_SHA256
            and metadata.get('source_reference') == source and metadata.get('execution_source') == execution
            and metadata.get('backbone_tensor_sha256') == identity['backbone_tensor_sha256']
            and metadata.get('backbone_updates') == 0 and metadata.get('head_updates') == expected_updates
            and metadata.get('completed_epoch') == 3 and metadata.get('frozen_R_exact_cache_replay') is True
            and metadata.get('optimizer_updates_this_evaluation') == 0 and metadata.get('test_evaluated') is False
            and metadata.get('archive_sha256') == expected['archive_sha256'], 'Probe producer metadata differs')
        replay = metadata.get('frozen_R_replay', {})
        require(isinstance(replay, dict) and replay.get('exact_equal') is True,
            'Strict frozen replay is missing')
        roles[selection] = metadata.get('head_roles', {})
        require(set(roles[selection]) == set(ARMS), 'Missing head roles')
        for arm, role in roles[selection].items():
            epoch = role.get('epoch')
            require(role.get('selection') == selection and role.get('policy') == arm
                and type(epoch) is int and 0 <= epoch <= 3
                and isinstance(role.get('head_tensor_sha256'), str), 'Invalid selected head role')
            validation_replay = metadata.get('validation_replay', {}).get(arm, {})
            require(validation_replay.get('passed') is True, 'Native validation replay missing: '+arm)
            if selection == 'last':
                require(epoch == 3 and role.get('checkpoint_sha256') == LAST_CHECKPOINT_SHA256,
                    'LAST head role differs')
            else:
                saved = summary.get('best', {}).get(arm, {})
                require(saved.get('epoch') == epoch and saved.get('checkpoint_sha256') == role.get('checkpoint_sha256'),
                    'BEST summary role differs')
                _capture(_child(directory, saved.get('checkpoint_file')), digests, role['checkpoint_sha256'])
        with np.load(archive_path, allow_pickle=False) as payload:
            archive = {key: payload[key].copy() for key in payload.files}
        y, _, _ = validate_archive(archive)
        require(y.shape == PRODUCTION_SHAPE, 'Expected the complete 837x12x496 development archive')
        require(np.array_equal(archive['sample_ids'], identity['data']['splits']['val']['selected_indices']),
            'Validation sample order differs from source data identity')
        for key, value in baseline.items():
            _same_array(archive[key], value, key)
        records = _metrics(archive)
        for arm in ARMS:
            selected_epoch = roles[selection][arm]['epoch']
            if selection == 'best':
                require(np.isclose(summary['best'][arm].get('mae_pooled'), records[arm]['mae_pooled'],
                    rtol=1e-10, atol=1e-10), 'BEST summary pooled MAE differs: '+arm)
            if selected_epoch > 0:
                expected_record = history[selected_epoch-1].get('validation', {}).get(arm, {})
                require(all(np.isclose(expected_record.get(key, np.nan), records[arm][key],
                    rtol=1e-10, atol=1e-10) for key in ('mae_pooled', 'rmse_pooled', 'mape_pooled')),
                    'Selected history pooled metrics differ: '+arm)
        archives[selection] = archive
    best_records, last_records = _metrics(archives['best']), _metrics(archives['last'])
    require(_screen_checks(best_records, last_records) == EXPECTED_CHECKS, 'Archive-derived failed gate differs')
    certificate = dict(schema='gw9_read_only_failed_probe_authentication_v1', probe_directory=str(directory),
        source_identity_sha256=SOURCE_IDENTITY_SHA256, data_identity=identity['data'],
        initializations=initial, head_roles=roles, source_reference=source,
        file_digests=digests, protected_directories=sorted(protected), evaluations=EXPECTED_EVALUATIONS,
        evaluation_directory=str(output), aggregate_records=dict(best=best_records, last=last_records),
        array_provenance={selection:dict(path=str(output/selection/'paired_predictions.npz'),
            sha256=EXPECTED_EVALUATIONS[selection]['archive_sha256'], shape=list(PRODUCTION_SHAPE),
            axis_order='sample,horizon,node', cases=list(CASES)) for selection in ('best','last')},
        screening_checks=EXPECTED_CHECKS, eligible_for_joint_screen=False,
        training_updates_this_analysis=0, inference_calls_this_analysis=0,
        checkpoint_deserializations=0, source_artifacts_written=False, test_evaluated=False)
    assert_unchanged(certificate)
    return archives['best'], archives['last'], certificate


def assert_unchanged(certificate):
    """Rehash every captured file, including opaque checkpoint bytes."""
    for filename, expected in certificate['file_digests'].items():
        require(sha256(filename) == expected, 'Authenticated source changed: '+filename)


def require_output_outside_sources(output, probe_dir, certificate):
    """Require a fresh output, resolving symlinks before protecting source roots."""
    candidate = Path(output).resolve()
    require(not candidate.exists(), 'Use a fresh analysis output directory')
    protected = {Path(probe_dir).resolve(), *(Path(path).resolve()
        for path in certificate['protected_directories'])}
    for directory in protected:
        require(candidate != directory and directory not in candidate.parents,
            'Analysis output must be outside immutable source directories: '+str(directory))
    # Protecting a source also rules out selecting its ancestor as the output.
    require(not any(candidate in directory.parents for directory in protected),
        'Analysis output must not contain an immutable source directory')
    return candidate
