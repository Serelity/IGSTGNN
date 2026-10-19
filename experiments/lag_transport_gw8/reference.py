"""Read-only certification of the completed GW7 and its recurrent BEST anchor."""
from pathlib import Path

from experiments.graph_interaction_gw7 import train as gw7
from experiments.graph_interaction_gw7 import evaluate as gw7_eval
from experiments.graph_interaction_gw7.reference import (
    read_json, require, verify_identity_digest, recertify as recertify_gw6,
    protected_paths as gw7_protected_paths)


SCHEMA = 'gw8_certified_gw7_recurrent_best_reference_v1'
SOURCE_IDENTITY_SHA256 = 'f2733dc591a8e6acf8bbdd0f3896395633e85aec3f33db2ee26287fa0354b796'
SOURCE_CHECKPOINT_SHA256 = '568ff5e0dcea220cb28e2e9bec794ef1894411e8dd8acd4bf427bfb2e425ca65'
SOURCE_ARM = 'R_CONTINUE'
SOURCE_EPOCH = 4


def protected_paths(reference):
    return {Path(reference['source_run']).resolve(),
        *gw7_protected_paths(reference['gw7_source_reference'])}


def _evaluation(run, identity, states, selection, explicit=None):
    candidates = [Path(explicit).resolve()/'metadata.json'] if explicit else sorted(run.glob('*/metadata.json'))
    matches = []
    for path in candidates:
        try:
            metadata = read_json(path)
            require(metadata.get('schema') == gw7_eval.PRODUCER_SCHEMA
                and metadata.get('status') == 'GW7_PREDICTION_PASS'
                and metadata.get('run_identity_sha256') == identity['identity_sha256']
                and metadata.get('source_reference') == identity['source_reference']
                and metadata.get('registered_arms') == list(gw7.ARMS)
                and metadata.get('checkpoint_selection') == selection
                and metadata.get('common_completed_new_epochs') == gw7.SCREEN_EPOCHS
                and metadata.get('best_selection_includes_stage_zero') is True
                and metadata.get('producer_source') == gw7_eval.producer_source()
                and metadata.get('engineering_only') is False
                and metadata.get('optimizer_updates') == 0
                and metadata.get('test_evaluated') is False,
                'GW7 paired evaluation certificate differs')
            for arm, state in states.items():
                require(metadata.get('checkpoint_roles', {}).get(arm) == state['_roles'][selection]
                    and metadata.get('validation_replay', {}).get(arm, {}).get('passed') is True,
                    'GW7 checkpoint role/native replay differs: '+arm)
            archive = path.parent/'paired_predictions.npz'
            require(gw7.base.sha256(archive) == metadata.get('archive_sha256'), 'GW7 archive changed')
            matches.append(dict(directory=str(path.parent.resolve()),
                metadata_sha256=gw7.base.sha256(path), archive_sha256=metadata['archive_sha256']))
        except (OSError, ValueError, KeyError):
            if explicit:
                raise
    require(bool(matches), 'Supply the completed GW7 '+selection+' evaluation')
    return matches[-1]


def certify(run, evaluations=None):
    run = Path(run).resolve()
    identity = read_json(run/'identity.json')
    verify_identity_digest(identity)
    require(identity.get('schema') == gw7.SCHEMA and identity.get('design') == gw7.DESIGN
        and identity.get('identity_sha256') == SOURCE_IDENTITY_SHA256
        and identity.get('arms') == list(gw7.ARMS)
        and identity.get('engineering_only') is False
        and identity.get('best_selection_includes_stage_zero') is True,
        'GW8 requires the existing certified GW7 five-epoch experiment')
    for name in gw7.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == gw7.base.sha256(gw7.REPO/name),
            'GW7 execution source differs: '+name)
    require(read_json(run/'source_reference.json') == identity['source_reference'],
        'GW7 source-reference file differs')
    _, gw4_weights, _ = recertify_gw6(identity['source_reference'])
    require(gw7.validate_diagnostic(Path(identity['diagnostic']['directory']), identity['source_reference'])
        == identity['diagnostic'], 'GW7 prerequisite layer diagnostic changed')
    budget = read_json(run/'budget.json')
    require(budget.get('target_completed_epochs') == gw7.SCREEN_EPOCHS
        and budget.get('extension_allowed') is False, 'GW7 completed budget differs')
    summary = read_json(run/'summary.json')
    require(summary.get('status') == 'GW7_TRAINING_COMPLETE'
        and summary.get('same_completed_new_training_budget') is True
        and summary.get('requested_new_epochs') == gw7.SCREEN_EPOCHS
        and summary.get('registered_arms') == list(gw7.ARMS)
        and summary.get('source_reference') == identity['source_reference']
        and summary.get('test_evaluated') is False, 'Complete all existing GW7 arms first')
    states, roles = {}, {}
    batches = len(identity['data']['splits']['train']['partition'])
    source_weights = None
    for arm in gw7.ARMS:
        directory = run/arm
        last_digest = gw7.base.sha256(directory/'last.pt')
        state = gw7.base.load_checkpoint(directory/'last.pt')
        gw7.validate_saved(state, identity, arm, directory, batches)
        require(state['completed_epoch'] == gw7.SCREEN_EPOCHS
            and summary['arms'][arm]['completed_epoch'] == gw7.SCREEN_EPOCHS
            and summary['arms'][arm]['last_model_tensor_sha256'] == state['model_tensor_sha256']
            and summary['arms'][arm]['best_checkpoint_sha256'] == state['best_checkpoint_sha256'],
            'GW7 completion summary differs: '+arm)
        best_weights, best_role = gw7_eval.selection(directory, state, 'best', arm)
        _, last_role = gw7_eval.selection(directory, state, 'last', arm)
        require(gw7.base.sha256(directory/'last.pt') == last_digest, 'GW7 LAST changed during read')
        state['_roles'] = dict(best=best_role, last=last_role)
        states[arm], roles[arm] = state, state['_roles']
        if arm == SOURCE_ARM:
            require(best_role['epoch'] == SOURCE_EPOCH
                and best_role['checkpoint_sha256'] == SOURCE_CHECKPOINT_SHA256,
                'GW8 source must remain the certified R_CONTINUE BEST epoch4 checkpoint')
            require(bool(best_weights) and all(name.startswith('anchor.') for name in best_weights),
                'Unexpected non-anchor tensors in GW7 R_CONTINUE BEST')
            source_weights = {name[len('anchor.'):]:value for name,value in best_weights.items()}
    archives = {role:_evaluation(run, identity, states, role, (evaluations or {}).get(role))
        for role in ('best', 'last')}
    source = states[SOURCE_ARM]
    row = source['history'][SOURCE_EPOCH-1]
    certificate = dict(schema=SCHEMA, source_run=str(run),
        source_identity_sha256=identity['identity_sha256'],
        source_identity_file_sha256=gw7.base.sha256(run/'identity.json'),
        source_budget_file_sha256=gw7.base.sha256(run/'budget.json'),
        source_summary_file_sha256=gw7.base.sha256(run/'summary.json'),
        gw7_source_reference=identity['source_reference'], source_diagnostic=identity['diagnostic'],
        completed_new_epochs=gw7.SCREEN_EPOCHS, checkpoint_roles=roles, archives=archives,
        source_initialization=dict(arm=SOURCE_ARM, selection='best', epoch=SOURCE_EPOCH,
            checkpoint_file=str(run/SOURCE_ARM/source['best_checkpoint_file']),
            checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            wrapper_model_tensor_sha256=roles[SOURCE_ARM]['best']['model_tensor_sha256'],
            model_tensor_sha256=gw7.base.tensor_tree_hash(source_weights),
            unwrapping='remove_exactly_one_anchor_prefix_from_every_tensor'),
        source_validation_metrics={key:row['validation_'+key] for key in ('mae','rmse','mape_fraction')},
        source_artifacts_written=False, new_training_updates=0, new_inference_calls=0)
    return certificate, identity, gw4_weights, source_weights


def recertify(saved):
    current, identity, gw4_weights, recurrent_weights = certify(saved['source_run'],
        {role:saved['archives'][role]['directory'] for role in ('best','last')})
    require(current == saved, 'Certified GW7 evidence changed since GW8 construction')
    return identity, gw4_weights, recurrent_weights
