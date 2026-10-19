"""Read-only GW13 source authentication; checkpoints stay opaque and untouched."""
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.trajectory_diagnostic_gw13 import model as old_model, protocol as old_protocol
from experiments.trajectory_diagnostic_gw13 import run as old_run


PINNED_MAES = {
    'audit': dict(SUM=13.333157028, ORIGIN=13.270295299, OWN4=13.242757894,
        CONTEXT=13.241865251, MAGNITUDE16=13.239526825, DIRECTION16=13.234936921,
        TRAJECTORY16=13.229434693),
    'validation': dict(SUM=13.368020244, ORIGIN=13.288735464, OWN4=13.246264357,
        CONTEXT=13.245614762, MAGNITUDE16=13.242892828, DIRECTION16=13.240356899,
        TRAJECTORY16=13.233809792),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _child(root, name):
    path = (root/name).resolve()
    require(path.parent == root and path.is_file(), 'Missing direct GW13 source file: '+name)
    return path


def assert_metrics(actual, expected, name):
    """Numeric reduction replay, never a relaxed prediction-tensor tolerance."""
    require(isinstance(actual, dict) and isinstance(expected, dict) and set(actual) == set(expected),
            'GW13 metric keys differ: '+name)
    for key, value in actual.items():
        saved = expected[key]
        if isinstance(value, dict):
            assert_metrics(value, saved, name+'/'+key)
        elif value is None:
            require(saved is None, 'GW13 empty metric differs: '+name+'/'+key)
        elif key == 'valid_count' or (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))):
            require(isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_))
                and type(saved) is int and int(value) == saved, 'GW13 count differs: '+name+'/'+key)
        else:
            require(isinstance(value, Real) and not isinstance(value, (bool,np.bool_))
                and not isinstance(saved, bool) and isinstance(saved, (int, float))
                and np.isfinite(saved) and np.isfinite(value)
                and np.isclose(value, saved, atol=1e-10, rtol=1e-12),
                'GW13 numeric reduction differs: '+name+'/'+key)


def authenticate(directory, certificate, cache_digests, cache, history_metadata):
    """Bind completed failed GW13 tables to certified SUM/data and replay records.

    No new parent or source model is fitted. The saved audit prefix and full
    training tables remain distinct. Returned models are JSON data, not Torch.
    """
    root = Path(directory).resolve()
    names = ('analysis.json', 'protocol.json', 'screening.json', 'audit_decision.json',
             'audit_model.json', 'model.json', 'fitted_before_validation.json')
    paths = {name:_child(root, name) for name in names}
    digests = {str(path):auth.sha256(path) for path in paths.values()}
    analysis = read_json(paths['analysis.json'])
    saved_protocol = read_json(paths['protocol.json'])
    require(analysis.get('schema') == old_run.SCHEMA and analysis.get('method') == old_run.METHOD
        and analysis.get('model_protocol') == old_model.PROTOCOL
        and analysis.get('source_identity_sha256') == auth.SOURCE_IDENTITY_SHA256
        and analysis.get('source_certificate') == certificate
        and analysis.get('history_metadata') == history_metadata
        and analysis.get('neural_inference_calls') == analysis.get('neural_optimizer_steps') == 0
        and analysis.get('test_evaluated') is False, 'Expected the returned immutable GW13 diagnostic')
    execution = {str(old_run.REPO/name):auth.sha256(old_run.REPO/name) for name in old_run.SOURCE_FILES}
    initial_digests = dict(cache_digests, **execution)
    expected_inputs = dict(initial_digests, **history_metadata['file_digests'])
    require(saved_protocol == dict(schema=old_run.SCHEMA, method=old_run.METHOD, model=old_model.PROTOCOL,
        source_identity_sha256=auth.SOURCE_IDENTITY_SHA256, cache_digests=initial_digests,
        execution_source=execution, expected_data_identity=certificate['data_identity']),
        'GW13 declared method or cache execution provenance differs')
    require(analysis.get('input_digests') == expected_inputs, 'GW13 input fingerprints differ')
    require(analysis.get('cache_metadata_sha256') in cache_digests.values(), 'GW13 cache certificate differs')
    require(analysis.get('audit_model_sha256') == digests[str(paths['audit_model.json'])]
        and analysis.get('model_sha256') == digests[str(paths['model.json'])], 'GW13 model bytes differ')
    require(read_json(paths['fitted_before_validation.json']) == dict(
        model_sha256=analysis['model_sha256'], protocol_sha256=digests[str(paths['protocol.json'])],
        validation_labels_used_for_fit=False), 'GW13 model was not committed before scoring')
    audit = analysis.get('audit', {})
    validation = analysis.get('validation')
    require(isinstance(validation, dict), 'The returned GW13 must have completed development scoring')
    audit_decision = old_protocol.audit_screen(audit.get('records'), audit.get('coverage'))
    require(audit_decision == analysis.get('audit_decision') == read_json(paths['audit_decision.json'])
        and audit_decision['audit_passed'] is True, 'GW13 audit decision differs')
    decision = old_protocol.validation_screen(validation.get('records'), validation.get('groups'), audit_decision)
    require(decision == analysis.get('screening') == read_json(paths['screening.json'])
        and decision['trajectory_information_screen_passed'] is False
        and decision['eligible_for_architecture_design'] is False
        and decision['candidate_for_confirmation'] is False, 'Keep the original failed GW13 screen unchanged')
    models = {phase:old_model.validate_model(read_json(paths[name])) for phase,name in
              (('audit', 'audit_model.json'), ('validation', 'model.json'))}
    datasets = dict(audit=old_run.slice_data(cache['train'], old_run.SCORE_START, old_run.TRAIN_COUNT),
                    validation=cache['val'])
    for phase, fitted in models.items():
        fit_data = old_run.slice_data(cache['train'], 0, old_run.FIT_STOP) if phase == 'audit' else cache['train']
        expected_ids = hashlib.sha256(fit_data['sample_ids'].astype('<u8').tobytes()).hexdigest()
        require(fitted['nodes'] == 496 and fitted['horizon'] == 12
            and fitted['fit_sample_count'] == len(fit_data['sample_ids'])
            and fitted['fit_valid_count'] == int(fit_data['valid'].sum())
            and fitted['fit_sample_ids_sha256'] == expected_ids,
            'GW13 audit/full fit sample or mask identity differs: '+phase)
        # Reuse the original reduction implementation, without any neural call.
        replay = old_run.score(fitted, datasets[phase])
        assert_metrics(replay['records'], analysis[phase]['records'], phase+'/records')
        assert_metrics(replay['groups'], analysis[phase]['groups'], phase+'/groups')
        fresh_coverage = old_model.coverage(fitted, datasets[phase]['features'], datasets[phase]['valid'])
        require(fresh_coverage == analysis[phase]['coverage'], 'GW13 support replay differs: '+phase)
        for arm, expected in PINNED_MAES[phase].items():
            require(abs(replay['records'][arm]['mae']-expected) <= 5.1e-10,
                    'GW13 MAE differs from the returned result: '+phase+'/'+arm)
    require(analysis['cache_metadata_sha256'] == next(value for path,value in cache_digests.items()
        if Path(path).name == 'cache.json'), 'GW13 selected a different cache certificate')
    digests.update(expected_inputs)
    return dict(directory=str(root), analysis=analysis, models=models, datasets=datasets,
        digests=digests, source_screen_unchanged=True, model_replay_passed=True,
        new_neural_inference_calls=0, new_source_fits=0, test_evaluated=False)


def assert_unchanged(reference):
    for filename, expected in reference['digests'].items():
        require(auth.sha256(filename) == expected, 'Immutable GW13 source changed: '+filename)
