"""Read-only authentication of the completed GW6 four-arm, 20-epoch evidence."""
from pathlib import Path

from experiments.graph_memory_gw6 import train as gw6
from experiments.graph_memory_gw6 import evaluate as gw6_eval
from experiments.graph_memory_gw6.reference import (
    read_json, require, verify_identity_digest, recertify as recertify_gw5)


SOURCE_EPOCHS = 20
SCHEMA = 'gw7_certified_recurrent20_reference_v1'


def protected_paths(reference):
    paths = {Path(reference['source_run']).resolve()}
    paths.update(gw6.protected_run_paths(reference['gw6_source_reference']))
    return paths


def validate_diagnostic(directory, reference):
    directory = Path(directory).resolve()
    metadata = read_json(directory/'metadata.json')
    summary = read_json(directory/'summary.json')
    for value in (metadata, summary):
        require(value.get('status') == 'GW6_LAYER_DIAGNOSTIC_COMPLETE'
            and value.get('source_run_identity_sha256') == reference['source_identity_sha256']
            and value.get('source_run') == reference['source_run']
            and value.get('reconstruction_pass') is True
            and value.get('test_evaluated') is False
            and value.get('optimizer_updates') == 0,
            'Finish the matching read-only GW6 layer diagnostic before GW7 training')
        require(value.get('source_budget_epochs') == SOURCE_EPOCHS,
            'Diagnostic must use the completed 20-epoch GW6 budget')
        require(value.get('archive_prediction_replay_pass') is True
            and value.get('model_tensors_unchanged') is True and value.get('checkpoint_writes')==0
            and value.get('bn_recalibration') is False,
            'Diagnostic must replay archived predictions without changing source tensors')
        require(value.get('input_manifest_sha256')==gw6.base.sha256(directory/'input_manifest.json'),
            'Diagnostic input manifest digest differs')
        sources=value.get('producer_source',{})
        require(bool(sources) and all(sources.get(name)==gw6.base.sha256(gw6.REPO/name)
            for name in ('experiments/graph_memory_gw6/diagnose_layers.py',
                'experiments/graph_memory_gw6/diagnose_drift.py',
                'experiments/graph_memory_gw6/evaluate.py',
                'experiments/graph_memory_gw6/model.py')),
            'Diagnostic producer source differs')
        for role in ('best','last'):
            actual=value.get('checkpoint_roles',{}).get(role,{})
            expected=reference['checkpoint_roles']['JOINT_MEMORY'][role]
            require(all(actual.get(key)==expected[key] for key in
                ('selection','epoch','policy','checkpoint_sha256','model_tensor_sha256')),
                'Diagnostic JOINT checkpoint role differs: '+role)
    manifest=read_json(directory/'input_manifest.json')
    require(manifest.get('source_run_identity_sha256')==reference['source_identity_sha256']
        and manifest.get('diagnostic_script_sha256')==gw6.base.sha256(
            gw6.REPO/'experiments/graph_memory_gw6/diagnose_layers.py')
        and manifest.get('execution_source')=={name:gw6.base.sha256(gw6.REPO/name)
            for name in gw6.EXECUTION_FILES},'Diagnostic source or input manifest differs')
    return dict(directory=str(directory), metadata_sha256=gw6.base.sha256(directory/'metadata.json'),
        summary_sha256=gw6.base.sha256(directory/'summary.json'),
        input_manifest_sha256=gw6.base.sha256(directory/'input_manifest.json'),
        reconstruction_pass=True, causal_mechanism_claimed=False)


def _evaluation(run, identity, states, role, explicit=None):
    paths = [Path(explicit).resolve()/'metadata.json'] if explicit else sorted(run.glob('*/metadata.json'))
    matches = []
    for path in paths:
        try:
            value = read_json(path)
            require(value.get('schema') == gw6_eval.PRODUCER_SCHEMA
                and value.get('status') == 'GW6_PREDICTION_PASS'
                and value.get('run_identity_sha256') == identity['identity_sha256']
                and value.get('source_reference') == identity['source_reference']
                and value.get('registered_arms') == list(gw6.DEFAULT_ARMS)
                and value.get('stage') == 'joint'
                and value.get('checkpoint_selection') == role
                and value.get('common_completed_new_epochs') == SOURCE_EPOCHS
                and value.get('producer_source') == gw6_eval.producer_source()
                and value.get('engineering_only') is False
                and value.get('test_evaluated') is False,
                'GW6 evaluation certificate differs')
            for arm, state in states.items():
                expected = state['_roles'][role]
                actual = value.get('checkpoint_roles', {}).get(arm, {})
                require(all(actual.get(key) == item for key,item in expected.items()),
                    'GW6 evaluation checkpoint role differs: '+arm)
                require(value.get('validation_replay', {}).get(arm, {}).get('passed') is True,
                    'GW6 native validation replay failed: '+arm)
            archive = path.parent/'paired_predictions.npz'
            require(gw6.base.sha256(archive) == value.get('archive_sha256'), 'GW6 archive changed')
            matches.append(dict(directory=str(path.parent.resolve()),
                metadata_sha256=gw6.base.sha256(path), archive_sha256=value['archive_sha256']))
        except (OSError, ValueError, KeyError):
            if explicit:
                raise
    require(bool(matches), 'Missing completed 20-epoch GW6 '+role+' evaluation')
    return matches[-1]


def certify(run, evaluations=None):
    run = Path(run).resolve()
    identity = read_json(run/'identity.json')
    verify_identity_digest(identity)
    require(identity.get('schema') == gw6.SCHEMA and identity.get('design') == gw6.DESIGN
        and identity.get('stage') == 'joint' and identity.get('curriculum') == 'full'
        and identity.get('arms') == list(gw6.DEFAULT_ARMS)
        and identity.get('engineering_only') is False,
        'Expected the completed full-12, four-arm GW6 joint experiment')
    for name in gw6.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == gw6.base.sha256(gw6.REPO/name),
            'GW6 execution source differs: '+name)
    require(read_json(run/'source_reference.json') == identity['source_reference'],
        'GW6 source-reference file differs')
    _, gw4_weights, _ = recertify_gw5(identity['source_reference'])
    require(read_json(run/'budget.json')['target_completed_epochs'] == SOURCE_EPOCHS,
        'GW7 source must be the fixed completed 20-epoch GW6 budget')
    summary = read_json(run/'summary.json')
    require(summary.get('status') == 'GW6_TRAINING_COMPLETE'
        and summary.get('same_completed_new_training_budget') is True
        and summary.get('requested_new_epochs') == SOURCE_EPOCHS,
        'GW6 training summary is incomplete')
    states, roles = {}, {}
    batches = len(identity['data']['splits']['train']['partition'])
    for arm in gw6.DEFAULT_ARMS:
        directory = run/arm
        digest = gw6.base.sha256(directory/'last.pt')
        state = gw6.base.load_checkpoint(directory/'last.pt')
        gw6.validate_saved(state, identity, arm, directory, batches)
        require(state['completed_epoch'] == SOURCE_EPOCHS
            and summary['arms'][arm]['completed_epoch'] == SOURCE_EPOCHS
            and summary['arms'][arm]['last_model_tensor_sha256']==state['model_tensor_sha256']
            and summary['arms'][arm]['best_checkpoint_sha256']==state['best_checkpoint_sha256'],
            'Complete every existing GW6 arm: '+arm)
        weights, best_role = gw6_eval.selection(directory, state, 'best', arm)
        _, last_role = gw6_eval.selection(directory, state, 'last', arm)
        require(gw6.base.sha256(directory/'last.pt') == digest, 'GW6 LAST changed during read')
        state['_roles'] = {'best':best_role, 'last':last_role}
        states[arm], roles[arm] = state, state['_roles']
        del weights
    recurrent = states['RECURRENT_CONTROL']
    require(recurrent['best_epoch'] == SOURCE_EPOCHS
        and roles['RECURRENT_CONTROL']['best']['model_tensor_sha256'] ==
            roles['RECURRENT_CONTROL']['last']['model_tensor_sha256'],
        'Expected RECURRENT_CONTROL BEST=LAST at epoch 20')
    archives = {role:_evaluation(run,identity,states,role,(evaluations or {}).get(role))
        for role in ('best','last')}
    certificate = dict(schema=SCHEMA, source_run=str(run),
        source_identity_sha256=identity['identity_sha256'],
        source_identity_file_sha256=gw6.base.sha256(run/'identity.json'),
        source_budget_file_sha256=gw6.base.sha256(run/'budget.json'),
        source_summary_file_sha256=gw6.base.sha256(run/'summary.json'),
        gw6_source_reference=identity['source_reference'], completed_new_epochs=SOURCE_EPOCHS,
        checkpoint_roles=roles, archives=archives,
        source_initialization=dict(arm='RECURRENT_CONTROL',selection='last',epoch=SOURCE_EPOCHS,
            checkpoint_file=str(run/'RECURRENT_CONTROL'/'last.pt'),
            checkpoint_sha256=roles['RECURRENT_CONTROL']['last']['checkpoint_sha256'],
            model_tensor_sha256=roles['RECURRENT_CONTROL']['last']['model_tensor_sha256']),
        source_validation_metrics={key:recurrent['history'][-1]['validation_'+key]
            for key in ('mae','rmse','mape_fraction')},
        source_artifacts_written=False, new_training_updates=0, new_inference_calls=0)
    return certificate, identity, gw4_weights, recurrent['model_state']


def recertify(saved):
    current, identity, gw4, recurrent = certify(saved['source_run'],
        {key:saved['archives'][key]['directory'] for key in ('best','last')})
    require(current == saved, 'Certified GW6 evidence changed since GW7 construction')
    return identity, gw4, recurrent
