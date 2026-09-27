"""Train-only fixed-A path audit: normalization, ICSF, TIID and graph replay."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_matched_controls import sha256
from experiments.chronological.materialize_incident_branch import (
    FullPositiveDataset, MatchedCounterfactualDataset, atomic_npz,
    load_protocol as load_baseline_protocol,
)
from experiments.chronological.smoke import make_model, set_seed
from experiments.chronological.train import configure_determinism
from src.models.incident_response import forecast_clock_embeddings

PROTOCOL = Path(__file__).with_name('architecture_mechanism_audit_v12a.json')
PROTOCOL_SHA256 = '8334dfcdcfccfc9526e606ddec40536d284ca603070222d50bf4308674bbce08'
PATHS = ('off', 'norm_only', 'icsf_only', 'tiid_only', 'full', 'full_norm_graph')
CONTRASTS = {
    'normalization': {'norm_only': 1, 'off': -1},
    'icsf_given_normalization': {'icsf_only': 1, 'norm_only': -1},
    'tiid_given_normalization': {'tiid_only': 1, 'norm_only': -1},
    'tiid_given_icsf': {'full': 1, 'icsf_only': -1},
    'icsf_given_tiid': {'full': 1, 'tiid_only': -1},
    'full_vs_off': {'full': 1, 'off': -1},
    'icsf_tiid_interaction': {
        'full': 1, 'icsf_only': -1, 'tiid_only': -1, 'norm_only': 1},
    'dynamic_graph_replay_sensitivity': {'full': 1, 'full_norm_graph': -1},
}
REGIONS = ('all', 'candidate_h1_h3', 'candidate_h4_h6', 'candidate_h7_h12',
           'noncandidate_h1_h6', 'noncandidate_h7_h12') + tuple(
               f'all_h{h}' for h in range(1, 13))


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_protocol(path):
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError('v12a frozen protocol fingerprint changed')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_inputs(data_dir, primary_dir, secondary_dir, checkpoint, protocol):
    """Verify train payloads only; do not even hash val/test arrays."""
    baseline_path = PROTOCOL.with_name(protocol['baseline_protocol'])
    if sha256(baseline_path) != protocol['baseline_protocol_sha256']:
        raise ValueError('Baseline input-identity protocol changed')
    baseline = load_baseline_protocol(baseline_path)
    hashes = {}

    def checked(path, expected):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Input fingerprint differs: {path}')
        hashes[str(Path(path).resolve())] = actual

    for name, expected in protocol['model_source_sha256'].items():
        checked(REPO / name, expected)
    checked(checkpoint, baseline['checkpoint']['best_model_sha256'])
    data_dir = Path(data_dir)
    checked(data_dir / 'summary.json', baseline['positive_package']['summary_sha256'])
    checked(data_dir / 'context_manifest.json',
            baseline['positive_package']['context_manifest_sha256'])
    summary = json.loads((data_dir / 'summary.json').read_text())
    context = json.loads((data_dir / 'context_manifest.json').read_text())
    for name in ('train_flow.npy', 'train_manifest.csv', 'station_ids.npy', 'scaler.json'):
        checked(data_dir / name, summary['files'][name])
    for name in ('train_context.npz', 'adjacency.npy'):
        checked(data_dir / name, context['outputs'][name])
    for directory, key in ((primary_dir, 'primary_control_inputs'),
                           (secondary_dir, 'secondary_control_inputs')):
        for name, expected in baseline[key].items():
            if name == 'summary.json' or name.startswith('train_'):
                checked(Path(directory) / name, expected)
    return baseline, hashes


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture_icsf(module, history, incident, tod, dow):
    captured = {}

    def fusion_hook(_module, args):
        captured['semantic_attention'] = args[0][..., :1]

    def norm_hook(_module, args):
        captured['pre_norm'] = args[0]

    handles = [module.icsf_fusion_mlp.register_forward_pre_hook(fusion_hook),
               module.output_norm.register_forward_pre_hook(norm_hook)]
    try:
        enhanced, context = module(history, incident, None, tod, dow)
    finally:
        for handle in handles:
            handle.remove()
    return enhanced, context, captured


def tensor_difference(left, right, mask=None):
    delta = (left.detach().double() - right.detach().double()).abs()
    if mask is not None:
        delta = delta[mask.expand_as(delta)]
    if not delta.numel():
        return {'cells': 0, 'abs_mean': None, 'abs_max': None}
    return {'cells': delta.numel(), 'abs_mean': float(delta.mean()),
            'abs_max': float(delta.max())}


def synthetic_probe(model, seed):
    """Autograd is restricted to a CPU module copy and artificial, label-free inputs."""
    module = copy.deepcopy(model.icsf_module).cpu().eval().requires_grad_(True)
    rng = torch.Generator().manual_seed(seed)
    width = module.q_proj.in_features
    time_width = model.T_i_D_emb.shape[1]
    history = torch.randn(2, 12, 4, width, generator=rng)
    tod = torch.randn(2, time_width, generator=rng)
    dow = torch.randn(2, time_width, generator=rng)
    incident = {
        'report_age_minutes': torch.tensor([1., 4.]),
        'distances': torch.tensor([[[0., .9, 0.], [0., .2, 1.], [0., 0., 0.],
                                    [0., 0., 0.]]] * 2),
    }
    enhanced, _, captured = capture_icsf(module, history, incident, tod, dow)
    mask = incident['distances'].abs().sum(-1, keepdim=True) > 0
    injection = captured['pre_norm'] - history[:, -1]
    key = module.embed_incident_features(incident, tod, dow)
    expected_injection = module.v_proj(key)[:, None, :] * mask
    projection = torch.randn(enhanced.shape, generator=rng)
    named = dict(module.named_parameters())
    gradients = torch.autograd.grad((enhanced * projection).sum(),
                                    tuple(named.values()), allow_unused=True)
    grad_stats = {name: (None if grad is None else float(grad.abs().max()))
                  for name, grad in zip(named, gradients)}
    with torch.no_grad():
        changed_history = history + torch.randn(history.shape, generator=rng)
        _, _, changed_h = capture_icsf(module, changed_history, incident, tod, dow)
        changed_incident = {**incident, 'distances': incident['distances'].clone()}
        changed_incident['distances'][..., 1] *= .25
        _, _, changed_d = capture_icsf(module, history, changed_incident, tod, dow)
        zero_incident = {**incident, 'distances': torch.zeros_like(incident['distances'])}
        zero_enhanced, _, _ = capture_icsf(module, history, zero_incident, tod, dow)
    return {
        'scope': 'synthetic_ICSF_only_copy_no_Y_no_optimizer',
        'gradient_abs_max': grad_stats,
        'key_projection_can_still_learn_through_TIID_in_full_model': True,
        'semantic_attention_vs_mask': tensor_difference(
            captured['semantic_attention'], mask.to(history)),
        'pre_norm_injection_vs_mask_times_V': tensor_difference(injection, expected_injection),
        'injection_change_after_history_perturbation': tensor_difference(
            changed_h['pre_norm'] - changed_history[:, -1], injection),
        'injection_change_after_same_support_distance_perturbation': tensor_difference(
            changed_d['pre_norm'] - history[:, -1], injection),
        'disconnected_latest_state_change': tensor_difference(
            enhanced[:, -1], history[:, -1], ~mask),
        'zero_support_latest_state_change': tensor_difference(
            zero_enhanced[:, -1], history[:, -1]),
        'zero_support_vs_normalization_only': tensor_difference(
            zero_enhanced[:, -1], module.output_norm(history[:, -1])),
        'nonlatest_history_change': tensor_difference(enhanced[:, :-1], history[:, :-1]),
        'interpretation': 'Singleton attention is a mathematical property, not a new performance result. Float32 subtraction can leave roundoff in reconstructed injections.',
    }


def backbone(model, history, static, dynamic, node_u, node_d, tod, dow):
    diff, inherent = [], []
    for layer in model.layers:
        history, part_d, part_i = layer(history, dynamic, static, node_u, node_d, tod, dow)
        diff.append(part_d)
        inherent.append(part_i)
    return sum(diff) + sum(inherent)


@torch.inference_mode()
def path_predictions(model, x, incident, atol=1e-6, rtol=1e-6):
    if model.training or model._time_response_mode != 'fixed':
        raise ValueError('Path audit requires eval-mode fixed A')
    history, node_u, node_d, tod, dow = model._prepare_inputs(x)
    embedded = model.embedding(history)
    event_tod, event_dow = forecast_clock_embeddings(
        incident, model.T_i_D_emb, model.D_i_W_emb)
    enhanced, context, captured = capture_icsf(
        model.icsf_module, embedded, incident, event_tod, event_dow)
    normalized = embedded.clone()
    normalized[:, -1] = model.icsf_module.output_norm(embedded[:, -1])
    mask = incident['distances'].abs().sum(-1, keepdim=True) > 0
    expected_v = model.icsf_module.v_proj(model.icsf_module.embed_incident_features(
        incident, event_tod, event_dow))[:, None, :] * mask
    expected_pre_norm = embedded[:, -1] + expected_v
    diagnostics = {
        'semantic_attention_vs_mask': tensor_difference(
            captured['semantic_attention'], mask.to(embedded)),
        'pre_norm_vs_history_plus_mask_V': tensor_difference(
            captured['pre_norm'], expected_pre_norm),
        'normalization_non_candidate': tensor_difference(
            normalized[:, -1], embedded[:, -1], ~mask),
        'icsf_vs_norm_non_candidate': tensor_difference(
            enhanced[:, -1], normalized[:, -1], ~mask),
        'icsf_vs_norm_candidate': tensor_difference(
            enhanced[:, -1], normalized[:, -1], mask),
    }
    outputs, norm_graph = {}, None
    for name, values in (('off', embedded), ('norm_only', normalized),
                         ('icsf_only', enhanced)):
        static, dynamic = model._graph_constructor(
            node_embedding_u=node_u, node_embedding_d=node_d, history_data=values,
            time_in_day_feat=tod, day_in_week_feat=dow)
        hidden = backbone(model, values, static, dynamic, node_u, node_d, tod, dow)
        outputs[name] = model._decode_forecast(hidden, None)
        if name == 'norm_only':
            norm_graph = dynamic
            outputs['tiid_only'] = model._decode_forecast(hidden, context)
        if name == 'icsf_only':
            outputs['full'] = model._decode_forecast(hidden, context)
            graph_deltas = [tensor_difference(a, b) for a, b in zip(dynamic, norm_graph)]
            if len(dynamic) != len(norm_graph):
                raise ValueError('Graph replay support count changed')
            diagnostics['dynamic_graph_icsf_vs_norm'] = {
                'cells': sum(d['cells'] for d in graph_deltas),
                'abs_mean': (sum(d['abs_mean'] * d['cells'] for d in graph_deltas) /
                             sum(d['cells'] for d in graph_deltas)) if graph_deltas else None,
                'abs_max': max((d['abs_max'] for d in graph_deltas), default=None),
            }
            replay = backbone(model, enhanced, static, norm_graph, node_u, node_d, tod, dow)
            outputs['full_norm_graph'] = model._decode_forecast(replay, context)
    for name, event in (('off', None), ('full', incident)):
        native = model(x, incident_data=event)
        torch.testing.assert_close(outputs[name], native, atol=atol, rtol=rtol)
        diagnostics[f'native_{name}_replay'] = tensor_difference(outputs[name], native)
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise ValueError('Non-finite intervention output')
    return {name: outputs[name] for name in PATHS}, diagnostics


def region_masks(candidate, prediction):
    shape = prediction.shape
    if shape[1] != 12 or shape[-1] != 1:
        raise ValueError('Expected predictions [batch,12,nodes,1]')
    candidate = candidate[:, None, :, None].expand(shape)
    step = torch.arange(12, device=prediction.device)[None, :, None, None]
    masks = [torch.ones_like(candidate), candidate & (step < 3),
             candidate & (step >= 3) & (step < 6), candidate & (step >= 6),
             ~candidate & (step < 6), ~candidate & (step >= 6)]
    return masks + [(step == h).expand(shape) for h in range(12)]


def sample_statistics(predictions, target, valid, candidate):
    """Store per-sample sufficient sums; sensitivity includes invalid-Y cells."""
    predictions = {name: value.double() for name, value in predictions.items()}
    target = target.double()
    if valid.dtype != torch.bool or valid.shape != target.shape:
        raise ValueError('Explicit target validity mask required')
    if not torch.isfinite(target[valid]).all():
        raise ValueError('Non-finite valid target')
    masks = region_masks(candidate, predictions['full'])
    dim = (1, 2, 3)
    pred_counts = torch.stack([mask.sum(dim) for mask in masks], -1)
    valid_counts = torch.stack([(mask & valid).sum(dim) for mask in masks], -1)
    errors = []
    for name in PATHS:
        error = (predictions[name] - target).abs()
        errors.append(torch.stack([
            torch.where(mask & valid, error, 0.).sum(dim) for mask in masks], -1))
    absolute, signed, maxima = [], [], []
    for terms in CONTRASTS.values():
        delta = sum(predictions[name] * coefficient for name, coefficient in terms.items())
        absolute.append(torch.stack([
            torch.where(mask, delta.abs(), 0.).sum(dim) for mask in masks], -1))
        signed.append(torch.stack([
            torch.where(mask, delta, 0.).sum(dim) for mask in masks], -1))
        maxima.append(torch.stack([
            torch.where(mask, delta.abs(), 0.).amax(dim) for mask in masks], -1))
    arrays = {'prediction_counts': pred_counts, 'valid_counts': valid_counts,
              'absolute_error_sums': torch.stack(errors, 1),
              'delta_abs_sums': torch.stack(absolute, 1),
              'delta_signed_sums': torch.stack(signed, 1),
              'delta_abs_max': torch.stack(maxima, 1)}
    return {key: value.cpu().numpy() for key, value in arrays.items()}


def merge_diagnostics(totals, diagnostics):
    for name, item in diagnostics.items():
        current = totals.setdefault(name, {'cells': 0, 'abs_sum': 0., 'abs_max': None})
        if item['cells']:
            current['cells'] += item['cells']
            current['abs_sum'] += item['abs_mean'] * item['cells']
            current['abs_max'] = max(current['abs_max'] or 0., item['abs_max'])


def finish_diagnostics(totals):
    return {name: {'cells': item['cells'], 'abs_max': item['abs_max'],
                   'abs_mean': item['abs_sum'] / item['cells'] if item['cells'] else None}
            for name, item in totals.items()}


def summarize(arrays):
    result = {}
    for index, region in enumerate(REGIONS):
        n = int(arrays['prediction_counts'][:, index].sum())
        v = int(arrays['valid_counts'][:, index].sum())
        result[region] = {
            'prediction_cells': n, 'valid_target_cells': v,
            'descriptive_mae': {
                name: float(arrays['absolute_error_sums'][:, j, index].sum() / v) if v else None
                for j, name in enumerate(PATHS)},
            'prediction_sensitivity': {
                name: {
                    'abs_mean': float(arrays['delta_abs_sums'][:, j, index].sum() / n) if n else None,
                    'signed_mean': float(arrays['delta_signed_sums'][:, j, index].sum() / n) if n else None,
                    'abs_max': float(arrays['delta_abs_max'][:, j, index].max()) if n else None,
                } for j, name in enumerate(CONTRASTS)},
        }
    return result


def audit_cohort(model, dataset, batch_size, device, output_path, protocol, progress):
    collected, diagnostics, seen = {}, {}, 0
    for batch_index, batch in enumerate(DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=0)):
        x = batch['x'].to(device)
        incident = {key: value.to(device) for key, value in batch['incident'].items()}
        predictions, observed = path_predictions(
            model, x, incident, protocol['native_replay_atol_standardized'],
            protocol['native_replay_rtol'])
        # Inference and the path/graph interventions are complete before targets enter metrics.
        raw = {name: value * dataset.scaler['std'] + dataset.scaler['mean']
               for name, value in predictions.items()}
        arrays = sample_statistics(raw, batch['y_flow'].to(device),
                                   batch['y_valid'].to(device), batch['candidate_mask'].to(device))
        arrays.update({key: batch[key].numpy() for key in ('positive_sample_index', 'source_index')})
        for key, value in arrays.items():
            collected.setdefault(key, []).append(value)
        merge_diagnostics(diagnostics, observed)
        seen += len(x)
        if batch_index % protocol['progress_every_batches'] == 0 or seen == len(dataset):
            progress('cohort_progress', completed=seen, total=len(dataset))
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    if len(set(arrays['positive_sample_index'].tolist())) != len(dataset):
        raise ValueError('Duplicate or missing output sample identities')
    atomic_npz(output_path, **arrays, paths=np.asarray(PATHS),
               contrasts=np.asarray(tuple(CONTRASTS)), regions=np.asarray(REGIONS))
    return {'samples': seen, 'regions': summarize(arrays),
            'mechanisms_standardized': finish_diagnostics(diagnostics)}


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, protocol_path=PROTOCOL,
        device_name='cuda:0', batch_size=None, check=False):
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('v12a output or .partial exists; preserve it and use a new name')
    protocol = load_protocol(protocol_path)
    actual_batch_size = protocol['batch_size'] if batch_size is None else batch_size
    if actual_batch_size < 1:
        raise ValueError('Batch size must be positive')
    partial.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    identity = {'host': socket.gethostname(), 'pid': os.getpid(),
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'output': str(output.resolve()), 'engineering_check': check}

    def progress(stage, **fields):
        record = {**identity, 'stage': stage,
                  'elapsed_seconds': round(time.monotonic() - started, 3), **fields}
        write_json(partial / 'progress.json', record)
        print(json.dumps(record), flush=True)

    try:
        progress('verifying_inputs')
        baseline, inputs = verify_inputs(data_dir, primary_dir, secondary_dir, checkpoint, protocol)
        progress('inputs_verified')
        device = torch.device(device_name)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        configure_determinism(device)
        torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '3'))))
        set_seed(protocol['synthetic_seed'])
        model = make_model(Path(data_dir), baseline['expected_sensor_count'], device, 'fixed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        if sum(p.numel() for p in model.parameters()) != baseline['checkpoint']['parameters']:
            raise ValueError('Fixed-A parameter count changed')
        model.eval().requires_grad_(False)
        original_state = state_hash(model)
        progress('checkpoint_loaded')
        synthetic = synthetic_probe(model, protocol['synthetic_seed'])
        write_json(partial / 'synthetic.json', synthetic)
        progress('synthetic_probe_complete')
        results, outputs, compatibility = {}, {}, {}
        limit = protocol['check_samples_per_cohort'] if check else None
        if check:
            actual_batch_size = min(actual_batch_size, limit)
        for cohort in protocol['cohorts']:
            if cohort == 'incident_full':
                dataset = FullPositiveDataset(data_dir, 'train',
                    expected_count=baseline['expected_positive_samples']['train'],
                    expected_nodes=baseline['expected_sensor_count'], sample_limit=limit)
            else:
                dataset = MatchedCounterfactualDataset(data_dir, primary_dir, secondary_dir,
                    'train', cohort, expected_count=baseline['expected_common_triples']['train'],
                    expected_nodes=baseline['expected_sensor_count'], sample_limit=limit,
                    expected_frozen_only_pairs=baseline['candidate_mask_compatibility'][
                        'expected_frozen_only_pairs']['train'])
                compatibility[cohort] = dataset.candidate_mask_compatibility
            progress('cohort_started', cohort=cohort, samples=len(dataset))
            name = f'train_{cohort}_mechanisms.npz'
            results[cohort] = audit_cohort(model, dataset, actual_batch_size, device,
                partial / name, protocol,
                lambda stage, **fields: progress(stage, cohort=cohort, **fields))
            outputs[name] = {'sha256': sha256(partial / name), 'samples': len(dataset),
                             'bytes': (partial / name).stat().st_size}
            write_json(partial / f'{cohort}.json', results[cohort])
            progress('cohort_complete', cohort=cohort)
        if original_state != state_hash(model) or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Frozen checkpoint parameters, buffers or gradients changed')
        code_files = [Path(__file__), REPO / 'src/utils/chronological.py',
                      REPO / 'experiments/chronological/materialize_incident_branch.py',
                      REPO / 'experiments/chronological/smoke.py',
                      REPO / 'experiments/chronological/train.py']
        summary = {
            'status': 'ENGINEERING_CHECK_PASS' if check else 'ARCHITECTURE_MECHANISM_AUDIT_COMPLETE',
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_path),
            'engineering_check': check, 'full_cohort_evaluated': not check,
            'main_training_ready': False, 'model_training_performed': False,
            'optimizer_steps': 0, 'synthetic_gradient_computation_performed': True,
            'real_data_gradient_computation_performed': False,
            'checkpoint_state_unchanged': True, 'validation_arrays_read': False,
            'test_split_read': False, 'train_Y_used_for_descriptive_error': True,
            'independent_confirmation': False, 'checkpoint': baseline['checkpoint'],
            'paths': protocol['path_definitions'], 'contrasts': CONTRASTS,
            'synthetic': synthetic, 'results': results, 'candidate_mask_compatibility': compatibility,
            'inputs': inputs, 'outputs': outputs,
            'code_sha256': {str(p.relative_to(REPO)): sha256(p) for p in code_files},
            'environment': {**identity, 'python': sys.version, 'numpy': np.__version__,
                'torch': torch.__version__, 'device': str(device), 'threads': torch.get_num_threads(),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                                     text=True).strip()},
            'recommendation': 'ENGINEERING_ONLY' if check else protocol['decision'],
            'interpretation': protocol['interpretation'],
        }
        write_json(partial / 'summary.json', summary)
        progress('publishing')
        if output.exists():
            raise FileExistsError('Final output appeared during audit; refusing to overwrite')
        partial.rename(output)
        print(f'Saved v12a architecture audit: {output / "summary.json"}', flush=True)
        return summary
    except BaseException as error:
        write_json(partial / 'failure.json', {**identity, 'status': 'FAILED',
            'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'primary-control-dir', 'secondary-control-dir', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    run(args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.checkpoint,
        args.output, args.protocol, args.device, args.batch_size, args.check)


if __name__ == '__main__':
    main()
