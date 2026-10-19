"""Replay the failed GW11 LAST, then remove only its added adapter at inference.

This module never calls a training entry point or creates an optimizer. The
full jointly trained SUM source, including its original fitted readout head,
is used directly for OFF. Source parameter flags and all tensors stay intact.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from experiments.state_gate_gw11 import train as gw11
from experiments.state_gate_gw11 import ablation_auth as auth
from experiments.state_gate_gw11 import ablation_metrics
from experiments.state_gate_gw11.protocol import ARMS, EPOCHS, UPDATES_PER_ARM
from experiments.state_gate_gw11.reference import load_inputs, assert_unchanged as assert_reference_unchanged
from experiments.state_gate_gw11.mask_audit import validate_audit

SCHEMA = 'gw11_last_adapter_ablation_v1'
REPO, base = gw11.REPO, gw11.base
EXECUTION_FILES = tuple(dict.fromkeys((*gw11.EXECUTION_FILES,
    'experiments/state_gate_gw11/ablate_last.py',
    'experiments/state_gate_gw11/ablation_auth.py',
    'experiments/state_gate_gw11/ablation_metrics.py')))
TABLES = ('decomposition', 'by_horizon', 'by_node', 'by_flow', 'by_window')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execution_hashes():
    return {name: base.sha256(REPO/name) for name in EXECUTION_FILES}


def exact_replay(actual, expected, name):
    require(isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
        and actual.dtype == expected.dtype and actual.shape == expected.shape
        and np.isfinite(actual).all() and np.isfinite(expected).all()
        and np.array_equal(actual, expected), 'Exact prediction replay failed: '+name)
    return dict(passed=True, shape=list(actual.shape), dtype=str(actual.dtype),
        differing_points=0, tensor_bytes_sha256=hashlib.sha256(actual.tobytes(order='C')).hexdigest())


def validation_identity(observed, saved):
    require(set(observed) == set(saved), 'Validation release identity keys differ')
    require(all(observed[key] == saved[key] for key in saved if key != 'splits'),
        'Validation release, normalization or parser identity changed')
    require(set(observed['splits']) == {'val'} and set(saved['splits']) == {'train', 'val'}
        and observed['splits']['val'] == saved['splits']['val'],
        'Validation partition differs from the trained run')
    return True


def parameter_flags(model):
    return {name: bool(value.requires_grad) for name, value in model.named_parameters()}


def source_only(model):
    """Keep the learned source weights/head; bypass only the GW11 wrapper."""
    model.assert_cache_clear()
    source = model.source
    require(source.arm == 'SUM_CONTROL' and source.stage == 'joint' and source.head is not None,
        'OFF requires the complete jointly trained SUM source')
    require(all(value.requires_grad for value in model.parameters()),
        'Do not freeze parameters: their replay flags must remain unchanged')
    return source


def assert_model_unchanged(model, tensor_hash, flags):
    model.assert_cache_clear()
    require(base.tensor_tree_hash(model.state_dict()) == tensor_hash,
        'Inference changed a model tensor or buffer')
    require(parameter_flags(model) == flags, 'Inference changed parameter trainability flags')
    require(all(value.grad is None for value in model.parameters()), 'Inference created gradients')


def evaluate_pair(model, data, device, baseline, original_last, expected, arm):
    """Require ON cache replay before issuing any OFF prediction for this arm."""
    model.eval()
    before, flags = base.tensor_tree_hash(model.state_dict()), parameter_flags(model)
    row, on, on_timing, on_diagnostics = gw11.evaluate(model, data, device, baseline, arm+'__ON')
    replay = exact_replay(on, original_last['pred__'+arm], arm+' LAST ON')
    require(gw11.same_record(row, expected), 'LAST metrics differ from authenticated checkpoint history: '+arm)
    assert_model_unchanged(model, before, flags)
    off_model = source_only(model)
    off_row, off, off_timing, off_diagnostics = gw11.evaluate(
        off_model, data, device, baseline, arm+'__OFF')
    assert_model_unchanged(model, before, flags)
    control = exact_replay(off, on, 'SUM_CONTINUE ON/OFF') if arm == 'SUM_CONTINUE' else None
    return on, off, dict(arm=arm, last_replay=replay, sum_identity_control=control,
        model_tensor_sha256_before=before, model_tensor_sha256_after=before,
        trainability_flags_unchanged=True, all_parameters_require_grad=all(flags.values()),
        no_gradients_created=True, complete_source_includes_trained_SUM_head=True,
        on=dict(metrics=row, timing=on_timing, mechanisms=on_diagnostics),
        off=dict(metrics=off_row, timing=off_timing, mechanisms=off_diagnostics),
        intervention='call_complete_trained_model_source_without_GW11_adapter_wrapper',
        optimizer_updates=0, new_training_epochs=0)


def load_bound_checkpoint(path, certificate):
    path = Path(path).resolve()
    expected = certificate['file_digests'].get(str(path))
    require(expected is not None and base.sha256(path) == expected, 'Unbound checkpoint: '+str(path))
    saved = base.load_checkpoint(path)
    require(base.sha256(path) == expected, 'Checkpoint changed while loading: '+str(path))
    return saved


def write_csv(path, rows):
    require(bool(rows), 'Cannot write an empty diagnostic table: '+str(path))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'data-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('threads must be positive')
    for name in ('run_dir', 'data_dir', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def run(args):
    last, identity, certificate = auth.load_run(args.run_dir)
    sources = certificate['source_paths']
    baseline, old_certificate, cache, _, cache_digests, gw10ref = load_inputs(
        Path(sources['probe_dir']), Path(sources['cache_dir']), Path(sources['gw10_dir']))
    require(old_certificate == identity['source_certificate'] and gw10ref == identity['gw10_reference'],
        'Original GW9/GW10 provenance changed')
    require(base.sha256(Path(sources['cache_dir'])/'cache.json') == identity['cache_metadata_sha256'],
        'Original cache metadata changed')
    gate = validate_audit(Path(sources['audit_dir']), old_certificate, cache_digests, gw10ref)
    require(gate == identity['mask_gate'], 'Original mask audit gate changed')
    for key in ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support'):
        require(last[key].dtype == baseline[key].dtype and np.array_equal(last[key], baseline[key]),
            'GW11 LAST axes differ from certified GW9: '+key)
    exact_replay(last['pred__FROZEN_SUM'], baseline['pred__SUM_CONTROL'], 'FROZEN_SUM source cache')
    require(gw11.source_hashes() == identity['execution_source'], 'Original GW11 execution source changed')
    output = auth.require_output_outside_sources(args.output_dir, args.run_dir, certificate, args.data_dir)
    # Also protect all recursively certified GW9 source directories before mkdir.
    from experiments.branch_readout_gw9.readout_auth import require_output_outside_sources as protect_gw9
    protect_gw9(output, Path(sources['probe_dir']), old_certificate)
    output.mkdir(parents=True, exist_ok=False)
    execution = execution_hashes()
    try:
        source_identity, gw4, recurrent = gw11.recertify(old_certificate['source_reference'])
        runtime = base.configure_runtime(args.device, args.threads, identity['seed'])
        require(runtime == identity['runtime'], 'Ablation must use the original complete runtime configuration')
        modules = base.get_original_modules()
        old_identity = gw11.read_json(Path(sources['probe_dir'])/'identity.json')
        require(modules.source_fingerprint == source_identity['source_fingerprint'],
            'Original model source differs')
        data = base.load_data(args.data_dir, identity['config'], modules,
            logging.getLogger('gw11_last_ablation'), check=False, training=False)
        validation_identity(data.identity, identity['data'])
        require(np.array_equal(data.station_ids, last['node_ids']), 'Ablation station order changed')
        require(set(data.loaders) == {'val_loader'}, 'Only a validation loader is allowed')
        old_state = load_bound_checkpoint(Path(sources['probe_dir'])/'last.pt', old_certificate)
        gw11.gw9_probe.validate_saved(old_state, old_identity, Path(sources['probe_dir']), identity['train_batches'])
        head, role = gw11.gw9_probe.selected_role(old_state, Path(sources['probe_dir']), 'SUM_CONTROL', 'best')
        require(role == old_certificate['head_roles']['best']['SUM_CONTROL'], 'Certified SUM head role differs')
        del old_state, cache
        gc.collect()
        predictions, off_predictions, passes, roles = {}, {}, {}, {}
        for arm in ARMS:
            directory = args.run_dir/arm
            saved = load_bound_checkpoint(directory/'last.pt', certificate)
            gw11.validate_saved(saved, identity, arm, directory, identity['train_batches'])
            require(saved['completed_epoch'] == EPOCHS and saved['global_updates'] == UPDATES_PER_ARM,
                'Only the completed three-epoch LAST is eligible')
            last_role = certificate['last_metadata']['checkpoint_roles'][arm]
            require(last_role['epoch'] == EPOCHS
                and last_role['checkpoint_file'] == 'last.pt'
                and last_role['model_tensor_sha256'] == saved['model_tensor_sha256'],
                'LAST checkpoint role differs from original archive')
            base.configure_runtime(args.device, args.threads, identity['seed'])
            model = gw11.model_for_arm(data, modules, args.device, arm, gw4, recurrent, head)
            require(gw11.initialization(model) == identity['initializations'][arm],
                'Initial source, head or adapter construction changed: '+arm)
            model.load_state_dict(saved['model_state'], strict=True)
            expected = saved['history'][-1]['validation']
            require(gw11.same_record(expected, certificate['last_metadata']['records'][arm]),
                'LAST history does not match the original metadata')
            on, off, details = evaluate_pair(model, data, args.device, baseline, last, expected, arm)
            predictions[arm], off_predictions[arm], passes[arm], roles[arm] = on, off, details, last_role
            del model, saved
            gc.collect()
            if torch.device(args.device).type == 'cuda':
                torch.cuda.empty_cache()
        # Analyze only predictions that passed strict same-checkpoint replay.
        analyzed_archive = dict(last, **{'pred__'+arm: value for arm, value in predictions.items()})
        result = ablation_metrics.analyze(analyzed_archive, off_predictions)
        result.update(source_run_identity_sha256=identity['identity_sha256'],
            optimizer_updates=0, new_training_epochs=0, test_evaluated=False,
            original_screening_changed=False, selected_checkpoint='LAST_epoch_3')
        archived = {key: last[key] for key in ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')}
        archived.update(schema=np.array('gw11_last_adapter_ablation_archive_v1'),
            axis_order=np.array('sample,horizon,node'), pred__FROZEN_SUM=last['pred__FROZEN_SUM'])
        for arm in ARMS:
            archived['pred__'+arm+'__ON'] = predictions[arm]
            archived['pred__'+arm+'__OFF'] = off_predictions[arm]
        np.savez_compressed(output/'paired_predictions.npz', **archived)
        table_files = {}
        for name in TABLES:
            if name not in result:
                continue
            path = output/(name+'.csv')
            write_csv(path, result[name])
            table_files[name] = dict(file=path.name, sha256=base.sha256(path), rows=len(result[name]))
        report = ablation_metrics.render_report(result)
        (output/'report.md').write_text(report, encoding='utf-8')
        compact = {key: value for key, value in result.items() if key not in TABLES or key == 'decomposition'}
        compact['tables'] = table_files
        base.write_json(output/'analysis.json', compact)
        require(execution_hashes() == execution, 'Ablation execution source changed during inference')
        auth.assert_unchanged(certificate)
        gw11.assert_source_unchanged(old_certificate)
        assert_reference_unchanged(gw10ref)
        require(validate_audit(Path(sources['audit_dir']), old_certificate, cache_digests, gw10ref) == gate,
            'Original mask gate changed during diagnostic inference')
        metadata = dict(schema=SCHEMA, status='GW11_LAST_ABLATION_COMPLETE',
            source_run_identity_sha256=identity['identity_sha256'], source_certificate=certificate,
            execution_source=execution, runtime=runtime, validation_data=data.identity,
            checkpoint_roles=roles, passes=passes, complete_validation_passes=2*len(ARMS),
            complete_model_forwards=2*len(ARMS)*data.loaders['val_loader'].num_batch,
            parameter_or_buffer_mutations=0, trainability_flag_changes=0,
            optimizer_updates=0, new_training_epochs=0, test_evaluated=False,
            original_screening=certificate['screening'], original_screening_changed=False,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            files={path.name: base.sha256(path) for path in output.iterdir() if path.is_file()})
        base.write_json(output/'metadata.json', metadata)
        print(json.dumps(dict(status=metadata['status'], output_dir=str(output),
            complete_validation_passes=metadata['complete_validation_passes'],
            optimizer_updates=0, test_evaluated=False)), flush=True)
        return compact
    except BaseException as error:
        base.write_json(output/'failure.json', dict(schema=SCHEMA, status='GW11_LAST_ABLATION_FAILED',
            error_type=type(error).__name__, error=str(error), optimizer_updates=0,
            original_outputs_written=False, test_evaluated=False))
        raise


def main(argv=None):
    return run(parse_args(argv))


if __name__ == '__main__':
    main()
