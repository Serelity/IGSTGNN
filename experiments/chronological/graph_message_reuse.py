"""Read-only GW1 references for GW2; no model inference or optimizer steps."""
import hashlib
import json
from pathlib import Path

from experiments.chronological.original_igstgnn_runtime import CONFIG, sha256
from experiments.chronological import train_original_graph_window as gw1
from experiments.chronological import train_original_graph_message as gw2


REUSED_ARMS = ('ORIGINAL', 'STATIC')
NEW_ARMS = ('POST_LOCAL', 'POST_MESSAGE')
RUNTIME_KEYS = ('torch', 'numpy', 'cuda', 'threads', 'seed', 'dtype',
                'deterministic_algorithms', 'tf32', 'cudnn_benchmark')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sidecar(run):
    run = Path(run).resolve()
    return run.with_name(run.name+'_reuse')


def verify_identity_digest(identity):
    content = {k: v for k, v in identity.items() if k != 'identity_sha256'}
    require(hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            == identity.get('identity_sha256'), 'Source identity digest differs')


def compatible(source, target):
    require(source['training_config'] == target['training_config'],
            'GW1/GW2 training recipe or seed differs; do not reuse incompatible runs')
    require(source['data'] == target['data'], 'GW1/GW2 data, masks, partitions or sample order differs')
    require(source['source_fingerprint'] == target['source_fingerprint'], 'Native source differs')
    require(source['engineering_only'] == target['engineering_only'], 'Engineering/full runs cannot be mixed')
    for key in RUNTIME_KEYS:
        require(source['runtime'].get(key) == target['runtime'].get(key), 'Runtime differs: '+key)
    original = target['initializations']['ORIGINAL']
    for arm in REUSED_ARMS:
        for key in ('original_tensor_sha256', 'post_construction_rng_sha256'):
            require(source['initializations'][arm][key] == original[key], 'Native initialization/RNG differs: '+arm)


def validate_archive_certificate(metadata, identity, states, selection):
    require(metadata.get('schema') == 'original_igstgnn_graph_window_producer_v1'
            and metadata.get('status') == 'GW1_PREDICTION_PASS', 'A completed GW1 producer certificate is required')
    require(metadata.get('run_identity_sha256') == identity['identity_sha256'], 'GW1 archive belongs to another run')
    require(metadata.get('engineering_only') == identity['engineering_only'], 'GW1 archive engineering flag differs')
    require(metadata.get('source_fingerprint') == identity['source_fingerprint'], 'GW1 archive native source differs')
    require(metadata.get('common_completed_epochs') == states['ORIGINAL']['completed_epoch'], 'GW1 archive budget differs')
    for name in ('evaluate_original_graph_window.py', 'evaluate_unified_adapter_v03.py'):
        require(metadata.get('producer_source', {}).get(name) == sha256(gw1.REPO/'experiments/chronological'/name),
                'GW1 archive producer source differs: '+name)
    for arm in REUSED_ARMS:
        role = metadata.get('checkpoint_roles', {}).get(arm, {})
        state = states[arm]
        epoch = state['best_epoch'] if selection == 'best' else state['completed_epoch']
        expected = state['best_checkpoint_sha256'] if selection == 'best' else state['_last_file_sha256']
        require(role.get('selection') == selection and role.get('epoch') == epoch
                and role.get('checkpoint_sha256') == expected, 'GW1 archive checkpoint role differs: '+arm)
        require(metadata.get('validation_replay', {}).get(arm, {}).get('passed') is True,
                'Missing saved GW1 validation replay: '+arm)


def find_archive(run, identity, states, selection, explicit=None):
    candidates = [Path(explicit).resolve()] if explicit else sorted(run.glob('*/paired_predictions.npz'))
    matches = []
    for path in candidates:
        try:
            metadata_path = path.parent/'metadata.json'
            metadata = read_json(metadata_path)
            validate_archive_certificate(metadata, identity, states, selection)
            require(sha256(path) == metadata.get('archive_sha256'), 'GW1 archive digest differs')
            matches.append(dict(path=str(path), sha256=sha256(path),
                metadata_path=str(metadata_path), metadata_sha256=sha256(metadata_path)))
        except (ValueError, KeyError, OSError):
            if explicit:
                raise
    require(bool(matches), 'No certified GW1 '+selection+' prediction archive found under '+str(run)
            +'; preserve/use the existing completed evaluation directory, do not retrain baselines')
    return matches[-1]


def load_reference(run, epochs, target=None, archives=None):
    run = Path(run).resolve()
    identity_path = run/'identity.json'
    identity = read_json(identity_path)
    require(identity.get('schema') == gw1.SCHEMA and identity.get('design') == gw1.DESIGN,
            'Expected an existing GW1 run, not the mature baseline or a MIN run')
    verify_identity_digest(identity)
    for name in gw1.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == sha256(gw1.REPO/name), 'GW1 execution source differs: '+name)
    if target is not None:
        compatible(identity, target)
    batches = len(identity['data']['splits']['train']['partition'])
    states, evidence = {}, {}
    for arm in REUSED_ARMS:
        directory = run/arm
        last = directory/'last.pt'
        before = sha256(last)
        state = gw1.load_checkpoint(last)
        gw1.validate_saved(state, identity, arm, directory, batches)
        require(state['completed_epoch'] == epochs,
                'Reuse requires exactly the same completed budget: '+arm+' has '
                +str(state['completed_epoch'])+', requested '+str(epochs)+'. Do not compare unequal budgets')
        require(sha256(last) == before, 'GW1 checkpoint changed during read')
        state['_last_file_sha256'] = before
        states[arm] = state
        evidence[arm] = dict(last_checkpoint_sha256=before,
            best_checkpoint_file=state['best_checkpoint_file'], best_checkpoint_sha256=state['best_checkpoint_sha256'],
            completed_epoch=state['completed_epoch'], best_epoch=state['best_epoch'],
            global_updates=state['global_updates'], new_optimizer_updates=0,
            implementation=('native_original_558ab01' if arm == 'ORIGINAL' else
                'GW1_sender_static_mathematically_equivalent_not_same_GW2_reduction_kernel'))
    cached = {selection: find_archive(run, identity, states, selection,
                (archives or {}).get(selection)) for selection in ('best', 'last')}
    reference = dict(schema='gw2_read_only_gw1_reference_v1', source_run=str(run),
        source_identity_sha256=identity['identity_sha256'], source_identity_file_sha256=sha256(identity_path),
        completed_epochs=epochs, reused_arms=evidence, archives=cached,
        engineering_only=identity['engineering_only'], source_artifacts_written=False,
        caveat='STATIC_is_archived_GW1_reference_not_a_retrained_GW2_static_kernel')
    return reference, identity, states


def preflight(reference_identity, args, runtime):
    expected = dict(CONFIG, seed=args.seed, device=args.device)
    if args.check:
        expected['bs'] = 2
    require(reference_identity['training_config'] == expected, 'Requested GW2 recipe differs from GW1')
    require(reference_identity['engineering_only'] == args.check, 'Engineering/full reuse mismatch')
    for key in RUNTIME_KEYS:
        require(reference_identity['runtime'].get(key) == runtime.get(key), 'Requested runtime differs: '+key)
    if args.run_dir.exists():
        identity = read_json(args.run_dir/'identity.json')
        require(identity.get('schema') == gw2.SCHEMA and identity.get('design') == gw2.DESIGN, 'Not an existing GW2 run')
        compatible(reference_identity, identity)


def new_states(run, identity, epochs):
    states = {}
    batches = len(identity['data']['splits']['train']['partition'])
    for arm in NEW_ARMS:
        directory = Path(run)/arm
        state = gw2.load_checkpoint(directory/'last.pt')
        gw2.validate_saved(state, identity, arm, directory, batches)
        require(state['completed_epoch'] == epochs, 'Finish new arm at the same budget: '+arm)
        states[arm] = state
    return states


def reused_summary(run, reference, source_identity, old_states, epochs):
    identity = read_json(Path(run)/'identity.json')
    compatible(source_identity, identity)
    states = {**old_states, **new_states(run, identity, epochs)}
    summary = gw2.paired_summary(states, identity, epochs)
    summary.update(status='GW2_REUSE_TRAINING_PASS',
        primary_baseline='original_IGSTGNN_558ab01_reused_from_certified_GW1_same_budget',
        reused_reference=reference, newly_trained_arms=list(NEW_ARMS), reused_arms=list(REUSED_ARMS),
        candidate_initial_router_tensors_equal=False,
        candidate_initial_router_caveat='GW1_STATIC_has_different_inactive_registered_tensors; two_new_dynamic_arms_match',
        baseline_new_training_updates=0, baseline_new_inference_calls=0)
    for arm in REUSED_ARMS:
        summary['arms'][arm].update(reused=True, source_run=reference['source_run'],
            implementation=reference['reused_arms'][arm]['implementation'], new_optimizer_updates=0)
    for arm in NEW_ARMS:
        summary['arms'][arm].update(reused=False)
    ignored = {}
    for arm in REUSED_ARMS:
        path = Path(run)/arm/'last.pt'
        if path.exists():
            previous = gw2.load_checkpoint(path)
            ignored[arm] = dict(path=str(path), completed_epoch=previous['completed_epoch'],
                global_updates=previous['global_updates'],
                note='prior_GW2_baseline_work_preserved_but_not_used_as_reference_or_repeated')
    summary['existing_GW2_baseline_checkpoints_not_used'] = ignored
    return summary
