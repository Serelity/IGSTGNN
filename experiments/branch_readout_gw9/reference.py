"""Read-only certification of the completed GW8 recurrent BEST source.

The epoch and file digest are read from the authenticated completed run, never
guessed from a BEST/LAST metric table. Returned weights are the entire GW8 R
wrapper state; callers remove exactly one anchor prefix only when constructing
its GW6 inner anchor, then strictly load the unchanged full wrapper.
"""
from pathlib import Path

from experiments.lag_transport_gw8 import train as gw8
from experiments.lag_transport_gw8 import evaluate as gw8_eval
from experiments.lag_transport_gw8.reference import (
    read_json, require, verify_identity_digest, recertify as recertify_gw7,
    protected_paths as gw8_protected_paths)


SCHEMA = 'gw9_certified_gw8_recurrent_best_reference_v1'
SOURCE_IDENTITY_SHA256 = '61aa8b9eab14a6d17315dfdd16db0ea290929dfc883c058b59f01552addc7f09'
SOURCE_ARM = 'R_CONTINUE'
SOURCE_VALIDATION_MAE = 13.508353274


def protected_paths(reference):
    return {Path(reference['source_run']).resolve(),
        *gw8_protected_paths(reference['gw8_source_reference'])}


def _evaluation(run, identity, states, selection, explicit=None):
    candidates = [Path(explicit).resolve()/'metadata.json'] if explicit else sorted(run.glob('*/metadata.json'))
    matches = []
    for path in candidates:
        try:
            metadata = read_json(path)
            require(metadata.get('schema') == gw8_eval.PRODUCER_SCHEMA
                and metadata.get('status') == 'GW8_PREDICTION_PASS'
                and metadata.get('run_identity_sha256') == identity['identity_sha256']
                and metadata.get('source_reference') == identity['source_reference']
                and metadata.get('registered_arms') == list(gw8.ARMS)
                and metadata.get('checkpoint_selection') == selection
                and metadata.get('common_completed_new_epochs') == gw8.SCREEN_EPOCHS
                and metadata.get('best_selection_includes_stage_zero') is True
                and metadata.get('producer_source') == gw8_eval.producer_source()
                and metadata.get('engineering_only') is False
                and metadata.get('optimizer_updates') == 0
                and metadata.get('test_evaluated') is False,
                'GW8 paired evaluation certificate differs')
            for arm, state in states.items():
                require(metadata.get('checkpoint_roles', {}).get(arm) == state['_roles'][selection]
                    and metadata.get('validation_replay', {}).get(arm, {}).get('passed') is True,
                    'GW8 checkpoint role/native replay differs: '+arm)
            for name in gw8_eval.HISTORICAL:
                require(metadata.get('validation_replay', {}).get(name, {}).get('passed') is True,
                    'GW9 requires the corrected GW8 historical replay records: '+name)
            require(isinstance(metadata.get('historical_replay_provenance'), dict),
                'GW8 corrected historical replay provenance missing')
            archive = path.parent/'paired_predictions.npz'
            require(gw8.base.sha256(archive) == metadata.get('archive_sha256'), 'GW8 archive changed')
            matches.append(dict(directory=str(path.parent.resolve()),
                metadata_sha256=gw8.base.sha256(path), archive_sha256=metadata['archive_sha256']))
        except (OSError, ValueError, KeyError):
            if explicit:
                raise
    require(bool(matches), 'Supply the completed corrected GW8 '+selection+' evaluation')
    return matches[-1]


def certify(run, evaluations=None):
    run = Path(run).resolve()
    identity = read_json(run/'identity.json')
    verify_identity_digest(identity)
    require(identity.get('schema') == gw8.SCHEMA and identity.get('design') == gw8.DESIGN
        and identity.get('identity_sha256') == SOURCE_IDENTITY_SHA256
        and identity.get('arms') == list(gw8.ARMS)
        and identity.get('engineering_only') is False
        and identity.get('best_selection_includes_stage_zero') is True,
        'GW9 requires the existing certified GW8 five-epoch experiment')
    for name in gw8.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == gw8.base.sha256(gw8.REPO/name),
            'GW8 execution source differs: '+name)
    require(read_json(run/'source_reference.json') == identity['source_reference'],
        'GW8 source-reference file differs')
    _, gw4_weights, _ = recertify_gw7(identity['source_reference'])
    budget = read_json(run/'budget.json')
    require(budget.get('target_completed_epochs') == gw8.SCREEN_EPOCHS
        and budget.get('extension_allowed') is False, 'GW8 completed budget differs')
    summary = read_json(run/'summary.json')
    require(summary.get('status') == 'GW8_TRAINING_COMPLETE'
        and summary.get('same_completed_new_training_budget') is True
        and summary.get('requested_new_epochs') == gw8.SCREEN_EPOCHS
        and summary.get('registered_arms') == list(gw8.ARMS)
        and summary.get('source_reference') == identity['source_reference']
        and summary.get('test_evaluated') is False, 'Complete all existing GW8 arms first')
    states, roles, source_weights = {}, {}, None
    batches = len(identity['data']['splits']['train']['partition'])
    for arm in gw8.ARMS:
        directory = run/arm
        last_digest = gw8.base.sha256(directory/'last.pt')
        state = gw8.base.load_checkpoint(directory/'last.pt')
        gw8.validate_saved(state, identity, arm, directory, batches)
        require(state['completed_epoch'] == gw8.SCREEN_EPOCHS
            and summary['arms'][arm]['completed_epoch'] == gw8.SCREEN_EPOCHS
            and summary['arms'][arm]['last_model_tensor_sha256'] == state['model_tensor_sha256']
            and summary['arms'][arm]['best_checkpoint_sha256'] == state['best_checkpoint_sha256'],
            'GW8 completion summary differs: '+arm)
        best_weights, best_role = gw8_eval.selection(directory, state, 'best', arm)
        _, last_role = gw8_eval.selection(directory, state, 'last', arm)
        require(gw8.base.sha256(directory/'last.pt') == last_digest, 'GW8 LAST changed during read')
        state['_roles'] = dict(best=best_role, last=last_role)
        states[arm], roles[arm] = state, state['_roles']
        if arm == SOURCE_ARM:
            require(bool(best_weights) and all(name.startswith('anchor.') for name in best_weights),
                'Unexpected non-anchor tensors in GW8 R_CONTINUE BEST')
            source_weights = best_weights
    archives = {role:_evaluation(run, identity, states, role, (evaluations or {}).get(role))
        for role in ('best', 'last')}
    source = states[SOURCE_ARM]
    epoch = source['best_epoch']
    row = source['stage_zero'] if epoch == 0 else source['history'][epoch-1]
    require(abs(row['validation_mae']-SOURCE_VALIDATION_MAE) <= 1e-5,
        'Certified GW8 R BEST native MAE differs from the returned experiment')
    certificate = dict(schema=SCHEMA, source_run=str(run),
        source_identity_sha256=identity['identity_sha256'],
        source_identity_file_sha256=gw8.base.sha256(run/'identity.json'),
        source_budget_file_sha256=gw8.base.sha256(run/'budget.json'),
        source_summary_file_sha256=gw8.base.sha256(run/'summary.json'),
        gw8_source_reference=identity['source_reference'],
        completed_new_epochs=gw8.SCREEN_EPOCHS, checkpoint_roles=roles, archives=archives,
        source_initialization=dict(arm=SOURCE_ARM, selection='best', epoch=epoch,
            checkpoint_file=str(run/SOURCE_ARM/source['best_checkpoint_file']),
            checkpoint_sha256=roles[SOURCE_ARM]['best']['checkpoint_sha256'],
            wrapper_model_tensor_sha256=roles[SOURCE_ARM]['best']['model_tensor_sha256'],
            model_tensor_sha256=gw8.base.tensor_tree_hash(source_weights),
            unwrapping='returned_weights_keep_full_GW8_wrapper; construction_removes_exactly_one_anchor_prefix_for_GW6_then_strictly_loads_full_GW8_wrapper'),
        source_validation_metrics={key:row['validation_'+key] for key in ('mae','rmse','mape_fraction')},
        source_artifacts_written=False, new_training_updates=0, new_inference_calls=0)
    return certificate, identity, gw4_weights, source_weights


def recertify(saved):
    current, identity, gw4_weights, recurrent_weights = certify(saved['source_run'],
        {role:saved['archives'][role]['directory'] for role in ('best','last')})
    require(current == saved, 'Certified GW8 evidence changed since GW9 construction')
    return identity, gw4_weights, recurrent_weights
