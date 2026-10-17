"""Certify existing GW2/GW1 artifacts without constructing baseline models."""
from pathlib import Path

from experiments.chronological import train_original_graph_message as gw2
from experiments.chronological import evaluate_original_graph_message as metrics
from experiments.chronological import evaluate_original_graph_message_reuse as cached_eval
from experiments.chronological.graph_message_reuse import (
    compatible, load_reference, read_json, require, sidecar, verify_identity_digest)


REUSED_ARMS = ('ORIGINAL', 'STATIC', 'POST_MESSAGE')
PRODUCERS = ('evaluate_original_graph_message_reuse.py', 'graph_message_reuse.py',
             'evaluate_original_graph_message.py')


def validate_certificate(metadata, identity, states, selection, epochs):
    require(metadata.get('schema') == 'original_igstgnn_graph_message_reuse_producer_v1'
        and metadata.get('status') == 'GW2_REUSE_PREDICTION_PASS',
        'Use a completed GW2 reuse evaluation, not a report-only reanalysis')
    require(metadata.get('run_identity_sha256') == identity['identity_sha256'], 'GW2 cache identity differs')
    require(metadata.get('engineering_only') == identity['engineering_only'], 'GW2 cache engineering flag differs')
    require(metadata.get('common_completed_epochs') == epochs, 'GW2 cache training budget differs')
    for name in PRODUCERS:
        require(metadata.get('producer_source', {}).get(name) == metrics.sha256(gw2.REPO/'experiments/chronological'/name),
                'GW2 cache producer differs: '+name)
    for arm in REUSED_ARMS:
        state = states[arm]
        epoch = state['best_epoch'] if selection == 'best' else state['completed_epoch']
        digest = state['best_checkpoint_sha256'] if selection == 'best' else state['_last_file_sha256']
        role = metadata.get('checkpoint_roles', {}).get(arm, {})
        require(role.get('selection') == selection and role.get('epoch') == epoch
                and role.get('checkpoint_sha256') == digest, 'GW2 cached checkpoint role differs: '+arm)
        require(metadata.get('validation_replay', {}).get(arm, {}).get('passed') is True,
                'GW2 cache replay not passed: '+arm)


def find_evaluation(run, identity, states, selection, epochs, explicit=None):
    paths = [Path(explicit).resolve()/'metadata.json'] if explicit else sorted(run.glob('*/metadata.json'))
    matches = []
    for path in paths:
        try:
            metadata = read_json(path)
            validate_certificate(metadata, identity, states, selection, epochs)
            archive = path.parent/'paired_predictions.npz'
            require(metrics.sha256(archive) == metadata.get('archive_sha256'), 'GW2 cached prediction digest differs')
            matches.append(dict(directory=str(path.parent.resolve()),
                metadata_sha256=metrics.sha256(path), archive_sha256=metadata['archive_sha256']))
        except (ValueError, KeyError, OSError):
            if explicit:
                raise
    require(bool(matches), 'No certified GW2 '+selection+' evaluation at the requested budget; '
        'point --gw2-'+selection+'-eval to its existing directory; do not repeat completed experiments')
    return matches[-1]


def certify(run, epochs, target=None, evaluations=None):
    run = Path(run).resolve()
    identity = read_json(run/'identity.json')
    require(identity.get('schema') == gw2.SCHEMA and identity.get('design') == gw2.DESIGN, 'Expected the existing GW2 run')
    verify_identity_digest(identity)
    require(read_json(run/'budget.json')['target_completed_epochs'] == epochs,
            'GW2 requested budget differs; finish and certify a common budget before comparison')
    for name in gw2.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == metrics.sha256(gw2.REPO/name), 'GW2 training source differs: '+name)
    if target is not None:
        compatible(identity, target)
    previous = read_json(sidecar(run)/'reference.json')
    old_ref, old_identity, states = load_reference(previous['source_run'], epochs, target=identity,
        archives={key: previous['archives'][key]['path'] for key in ('best', 'last')})
    require(old_ref == previous, 'GW2 saved GW1 reference changed')
    last = run/'POST_MESSAGE'/'last.pt'
    before = metrics.sha256(last)
    state = gw2.load_checkpoint(last)
    gw2.validate_saved(state, identity, 'POST_MESSAGE', last.parent,
        len(identity['data']['splits']['train']['partition']))
    require(state['completed_epoch'] == epochs, 'POST_MESSAGE must have exactly the same completed budget')
    require(metrics.sha256(last) == before, 'GW2 checkpoint changed during read')
    state['_last_file_sha256'] = before
    states['POST_MESSAGE'] = state
    for key in ('original_tensor_sha256', 'post_construction_rng_sha256'):
        require(state['initialization'][key] == identity['initializations']['ORIGINAL'][key],
                'GW2 POST_MESSAGE native initialization differs')
    if target is not None:
        compatible(old_identity, target)
    archives = {key: find_evaluation(run, identity, states, key, epochs,
                (evaluations or {}).get(key)) for key in ('best', 'last')}
    for saved in archives.values():
        metadata = read_json(Path(saved['directory'])/'metadata.json')
        require(metadata.get('reuse_reference') == old_ref, 'GW2 evaluation GW1 provenance differs')
    reference = dict(schema='gw3_certified_gw2_reference_v1', source_run=str(run),
        source_identity_sha256=identity['identity_sha256'],
        source_identity_file_sha256=metrics.sha256(run/'identity.json'),
        completed_epochs=epochs, archives=archives, gw1_reference=old_ref,
        reused_arms={arm: dict(completed_epoch=s['completed_epoch'], best_epoch=s['best_epoch'],
            last_checkpoint_sha256=s['_last_file_sha256'], best_checkpoint_sha256=s['best_checkpoint_sha256'],
            global_updates=s['global_updates']) for arm, s in states.items()},
        new_training_updates=0, new_inference_calls=0, source_artifacts_written=False,
        caveat='STATIC_is_GW1_prior_kernel_reference; POST_MESSAGE_is_GW2_not_capacity_matched_GW3')
    return reference, identity, states


def load_cached(reference, selection):
    import numpy as np
    saved = reference['archives'][selection]
    directory = Path(saved['directory'])
    require(metrics.sha256(directory/'metadata.json') == saved['metadata_sha256'], 'Reference metadata changed')
    require(metrics.sha256(directory/'paired_predictions.npz') == saved['archive_sha256'], 'Reference predictions changed')
    with np.load(directory/'paired_predictions.npz', allow_pickle=False) as payload:
        archive = dict(payload)
    metrics.validate_archive(archive, cached_eval.CASES, cached_eval.SCHEMA)
    require(metrics.sha256(directory/'paired_predictions.npz') == saved['archive_sha256'], 'Reference changed during read')
    return archive, read_json(directory/'metadata.json')
