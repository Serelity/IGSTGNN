"""GW4 read-only anchor provenance; no baseline model construction or inference."""
from pathlib import Path

from experiments.chronological import train_original_graph_message as base
from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
from experiments.chronological.receiver_match_reference import (
    certify as certify_gw2, load_cached as load_gw2_cached, read_json, require,
    verify_identity_digest)


def certify(run, evaluations=None):
    run = Path(run).resolve()
    epochs = read_json(run/'budget.json')['target_completed_epochs']
    reference, identity, states = certify_gw2(run, epochs, evaluations=evaluations)
    require(not identity['engineering_only'], 'GW4 adaptation requires the full certified GW2 run')
    state = states['POST_MESSAGE']
    path = run/'POST_MESSAGE'/state['best_checkpoint_file']
    digest = state['best_checkpoint_sha256']
    require(base.sha256(path) == digest, 'GW2 anchor checkpoint changed')
    weights = base.load_checkpoint(path)
    require(base.sha256(path) == digest, 'GW2 anchor changed while loading')
    best = next(row for row in state['history'] if row['epoch'] == state['best_epoch'])
    certificate = dict(schema='gw4_frozen_gw2_reference_v1', source_run=str(run),
        source_completed_epochs=epochs, selection='best', source_arm='POST_MESSAGE',
        source_checkpoint_file=str(path), source_checkpoint_sha256=digest,
        source_model_tensor_sha256=base.tensor_tree_hash(weights), source_best_epoch=state['best_epoch'],
        source_best_metrics={key: best['validation_'+key] for key in ('mae', 'rmse', 'mape_fraction')},
        gw2_reference=reference, source_artifacts_written=False,
        baseline_new_training_updates=0, baseline_new_inference_calls=0)
    return certificate, identity, weights


def compatible(source, target):
    for key in ('training_config', 'data', 'source_fingerprint', 'engineering_only'):
        require(source[key] == target[key], 'GW4 differs from its GW2 anchor: '+key)
    for key in RUNTIME_KEYS:
        require(source['runtime'].get(key) == target['runtime'].get(key), 'GW4 runtime differs: '+key)


def recertify(saved):
    evaluations = {key: saved['gw2_reference']['archives'][key]['directory'] for key in ('best', 'last')}
    current, identity, weights = certify(saved['source_run'], evaluations)
    require(current == saved, 'GW4 source certificate changed')
    return identity, weights


def load_cached(reference):
    # The anchor is always its BEST, even when evaluating adapted LAST states.
    return load_gw2_cached(reference['gw2_reference'], 'best')
