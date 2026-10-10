"""Inference-only P1 report -> capacity -> exchange -> forecast diagnosis.

The native IGSTGNN incident inputs are preserved. The intervention disables
ONLY the extra report conditioning of capacity, not the trained branch itself.
This is conditional checkpoint sensitivity, not a causal or trained P0 contrast.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_incident_capacity as training
from experiments.chronological.continue_incident_capacity import (
    PROTOCOL, array_digest, audit_arm, discover_run, regions_from_pack, verify_inputs, verify_sources,
)
from experiments.chronological.prepare_incident_corridors import discover_history
from experiments.chronological.smoke import make_model, set_seed
from src.models.incident_capacity_fusion import CapacityAugmentedIGSTGNN, IncidentCapacityBranch
from src.utils.capacity_fusion_inputs import OriginalCapacityInputs
from src.utils.capacity_training_inputs import CapacityDevelopmentInputs
from src.utils.incident_corridor import read_json, require, sha256, write_json, write_rows

# Fixed before observing the probe. Absolute tolerance is in original flow units.
REPLAY_ATOL, REPLAY_RTOL, REPLAY_MAE_ATOL = 1e-4, 1e-6, 1e-4
CHANGE_THRESHOLD = 1e-7  # Descriptive only; never a selection or significance rule.
ARTIFACTS = ('summary.json', 'last_checkpoint.pt', 'best_model.pt',
             'best_validation_predictions.npz', 'best_validation_metrics.json')


def snapshot_run(root):
    paths = [root/'identity.json', root/'paired_report.json']
    paths += [root/arm/name for arm in training.ARMS for name in ARTIFACTS]
    return {p.relative_to(root).as_posix(): sha256(p) for p in paths if p.is_file()}


def runtime_environment(device):
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                cuda=torch.version.cuda, device=str(device), host=platform.node(),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), threads=torch.get_num_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(), tf32=False)


def forward(model, batch, scaler, enabled):
    require(not any(m.training for m in model.modules()), 'Probe requires every module in eval mode')
    require(not torch.is_grad_enabled(), 'Probe requires inference without gradients')
    result = model(batch['x'], incident_data=batch['incident'],
                   capacity_inputs=batch['capacity_inputs'], incident_enabled=enabled, return_details=True)
    result['prediction'] = result['prediction']*scaler['std']+scaler['mean']
    return result


def tensor_tree_digest(value):
    if isinstance(value, dict):
        return {k: tensor_tree_digest(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return array_digest(value.detach().cpu().numpy())
    return value


def compare_replay(actual, expected):
    for key in ('target', 'valid', 'sample_indices'):
        require(actual[key].dtype == expected[key].dtype and
                np.array_equal(actual[key], expected[key]), 'Replay target/mask/order mismatch: '+key)
    a, b = actual['prediction'], expected['prediction']
    require(a.shape == b.shape and a.ndim == 4 and a.shape[-1] == 1
            and np.isfinite(a).all() and np.isfinite(b).all(), 'Replay prediction axes/values mismatch')
    delta = np.abs(a.astype(np.float64)-b)
    report = dict(prediction_abs_max=float(delta.max()), atol=REPLAY_ATOL, rtol=REPLAY_RTOL,
                  matching=bool(np.allclose(a, b, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)))
    require(report['matching'], 'ON replay differs from saved prediction; OFF pass blocked: '+str(report))
    return report


def input_regions(association, edge_index, nodes):
    """Report endpoints and undirected connected-component reachability.

    Weak connectivity includes upstream receiving interactions. It is a broad
    structural envelope, not observed impact, a time radius or a travel path.
    """
    association, edge_index = np.asarray(association), np.asarray(edge_index)
    require(association.ndim == 2 and edge_index.shape == (2, association.shape[1])
            and np.isfinite(association).all() and ((association >= 0) & (association <= 1)).all(),
            'Invalid association/edge axes')
    require(((edge_index >= -1) & (edge_index < nodes)).all(), 'Unknown graph endpoint')
    src, dst = edge_index
    internal = (src >= 0) & (dst >= 0)
    require(not (association[:, ~internal] > 0).any(), 'Reports must target internal edges')
    direct = np.zeros((len(association), nodes), bool)
    for edge in np.flatnonzero(internal):
        support = association[:, edge] > 0
        direct[:, src[edge]] |= support
        direct[:, dst[edge]] |= support
    parent = list(range(nodes))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edge_index[:, internal].T:
        parent[find(int(a))] = find(int(b))
    labels = np.array([find(i) for i in range(nodes)])
    potential = np.zeros_like(direct)
    for label in np.unique(labels):
        member = labels == label
        potential[:, member] = direct[:, member].any(1)[:, None]
    return dict(report_associated_nodes=direct, no_direct_report_association_nodes=~direct,
                potential_propagation_only_nodes=potential & ~direct,
                outside_potential_propagation_nodes=~potential,
                report_supported_samples=np.broadcast_to(direct.any(1)[:, None], direct.shape),
                no_report_supported_samples=np.broadcast_to(~direct.any(1)[:, None], direct.shape))


def paired_metric(on, off, target, valid, mask, weights=None):
    """Horizon-macro pooled MAE, paired on identical observations; empty is None."""
    require(on.shape == off.shape == target.shape == valid.shape and on.ndim == 4
            and on.shape[-1] == 1 and valid.dtype == bool, 'Paired metric axes mismatch')
    require(np.isfinite(on).all() and np.isfinite(off).all()
            and np.isfinite(target[valid]).all(), 'Nonfinite paired observations')
    mask = np.asarray(mask)
    require(mask.dtype == bool and mask.shape in ((on.shape[2],), (on.shape[0], on.shape[2])),
            'Region axes mismatch')
    if mask.ndim == 1:
        mask = np.broadcast_to(mask, (on.shape[0], on.shape[2]))
    selected = valid & mask[:, None, :, None]
    weights = np.ones(on.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    require(weights.shape == (on.shape[0],) and np.isfinite(weights).all() and (weights > 0).all(),
            'Invalid row weights')
    w = weights[:, None, None, None]
    count = (selected*w).sum((0, 2, 3))
    if not (count > 0).all():
        return None
    values = []
    for prediction in (on, off):
        errors = np.where(selected, np.abs(prediction.astype(np.float64)-target), 0)
        values.append((errors*w).sum((0, 2, 3))/count)
    change = np.where(selected, np.abs(off.astype(np.float64)-on), 0)
    return dict(on_mae_macro=float(values[0].mean()), off_mae_macro=float(values[1].mean()),
                off_minus_on_mae=float((values[1]-values[0]).mean()),
                per_horizon_on_mae=values[0].tolist(), per_horizon_off_mae=values[1].tolist(),
                per_horizon_off_minus_on=(values[1]-values[0]).tolist(),
                valid_count=int(selected.sum()), per_horizon_valid_count=selected.sum((0, 2, 3)).tolist(),
                per_horizon_weighted_count=count.tolist(),
                prediction_abs_change_mean=float((change*w).sum()/count.sum()),
                prediction_abs_change_max=float(change.max()))


def verify_intervention(on, off, restored, graph):
    for name in ('initial', 'hidden', 'summary', 'available', 'lag_intervals', 'history_coefficients',
                 'boundary_coefficients', 'boundary_demand', 'report_association'):
        require(torch.equal(on[name], off[name]), 'Report intervention changed history/boundary: '+name)
    require(torch.equal(off['coefficients'], off['history_coefficients']), 'OFF is not history-only capacity')
    unsupported = on['report_association'] == 0
    require(torch.equal(on['coefficients'][unsupported], on['history_coefficients'][unsupported]),
            'Report changed unsupported capacity coefficients')
    mask = unsupported[:, None, :, None].expand_as(on['capacity'])
    require(torch.equal(on['capacity'][mask], off['capacity'][mask]), 'Report changed unsupported capacities')
    require(tensor_tree_digest(on) == tensor_tree_digest(restored), 'Final ON restoration failed')
    require(on['rollout']['kind'] == off['rollout']['kind'] == 'capacity_limited', 'Expected P1 exchange operator')
    for result in (on, off):
        require((result['forecast_delta'][:, :, ~graph.operator_mask] == 0).all(),
                'Branch projection escaped operator mask')
        require(all(torch.isfinite(result[k]).all() for k in ('prediction', 'capacity', 'coefficients', 'forecast_delta')),
                'Nonfinite intervention result')


class PathwayTotals:
    """Stream batch reductions, retaining curves instead of full rollout tensors."""
    def __init__(self):
        self.sums, self.counts, self.maxima, self.changed = {}, {}, {}, {}
        self.association_sum = None
        self.flags, self.direct = {}, {}

    def update(self, on, off):
        fields = dict(coefficients=(on['coefficients'], off['coefficients'], (0,)),
                      capacity=(on['capacity'], off['capacity'], (0, 3)),
                      edge_flux=(on['rollout']['edge_flux'], off['rollout']['edge_flux'], (0, 3)),
                      states=(on['rollout']['states'], off['rollout']['states'], (0, 3)),
                      forecast_delta=(on['forecast_delta'], off['forecast_delta'], (0, 3)))
        for name, (a, b, axes) in fields.items():
            delta = b-a
            values = dict(on=a, off=b, off_minus_on=delta, abs_change=delta.abs())
            count = int(np.prod([a.shape[i] for i in axes]))
            self.counts[name] = self.counts.get(name, 0)+count
            for label, value in values.items():
                key = name+'_'+label
                reduced = value.sum(axes, dtype=torch.float64).cpu().numpy()
                self.sums[key] = self.sums.get(key, 0)+reduced
            self.maxima[name] = max(self.maxima.get(name, 0.), float(delta.abs().max()))
            self.changed[name] = self.changed.get(name, 0)+int((delta.abs() > CHANGE_THRESHOLD).sum())
            if name in ('coefficients', 'capacity', 'edge_flux'):
                support = on['report_association'] > 0
                support = (support[..., None] if name == 'coefficients' else support[:, None, :, None]).expand_as(a)
                positions = int(support.sum())
                old = self.direct.get(name, dict(positions=0, abs_change_sum=0., abs_change_max=None))
                old['positions'] += positions
                old['abs_change_sum'] += float(delta.abs()[support].sum(dtype=torch.float64))
                if positions:
                    old['abs_change_max'] = max(old['abs_change_max'] or 0., float(delta.abs()[support].max()))
                self.direct[name] = old
        association = on['report_association'].double().sum(0).cpu().numpy()
        self.association_sum = association if self.association_sum is None else self.association_sum+association
        supported = (on['report_association'] > 0)[:, None, :, None]
        for mode, result in (('on', on), ('off', off)):
            for flag in ('capacity_limited', 'receiving_limited', 'sending_bid_limited'):
                value = result['rollout'][flag]
                for scope, mask in (('all_recorded_edges', torch.ones_like(value)),
                                    ('direct_report_edges', supported.expand_as(value))):
                    key = mode+'_'+flag+'_'+scope
                    old = self.flags.get(key, [0, 0])
                    self.flags[key] = [old[0]+int((value & mask).sum()), old[1]+int(mask.sum())]

    def result(self, samples):
        curves, summary = {}, {}
        for name, count in self.counts.items():
            for label in ('on', 'off', 'off_minus_on', 'abs_change'):
                key = name+'_'+label
                curves[key+'_mean'] = self.sums[key]/count
            elements = count*self.sums[name+'_on'].size
            summary[name] = dict(abs_change_mean=float(curves[name+'_abs_change_mean'].mean()),
                                 abs_change_max=self.maxima[name], positions=elements,
                                 changed_fraction_above_threshold=self.changed[name]/elements)
        curves['report_association_mean'] = self.association_sum/samples
        summary['limitation_fractions'] = {k: dict(true_count=a, positions=b, fraction=a/b if b else None)
                                           for k, (a, b) in self.flags.items()}
        summary['change_threshold'] = CHANGE_THRESHOLD
        summary['direct_report_edges'] = {name: dict(positions=row['positions'],
            abs_change_mean=row['abs_change_sum']/row['positions'] if row['positions'] else None,
            abs_change_max=row['abs_change_max']) for name, row in self.direct.items()}
        summary['curve_weighting'] = 'original_rows_equal_weight; latent_channels_mean_except_four_coefficients'
        return summary, curves


def report_support(batch):
    reports = batch['capacity_inputs']['reports']
    matched = reports['present'] & (reports['weights'].sum(-1) > 0)
    count = reports['present'].sum(1).cpu().numpy()
    associated = matched.sum(1).cpu().numpy()
    age = torch.where(matched, reports['ages'], torch.inf).amin(1) if matched.shape[1] else torch.full_like(
        reports['ages'].sum(1), torch.inf)
    return count, associated, age.cpu().numpy()


def run_probe(model, inputs, indices, batch_size, probe_batch_size, device, scaler,
              saved_arrays, saved_metrics, regions, output_dir):
    """Replay ALL ON rows before any OFF inference; then paired ON/OFF/ON batches."""
    require(isinstance(model, CapacityAugmentedIGSTGNN) and model.branch.mode == 'capacity', 'Expected P1 model')
    require(not any(m.training for m in model.modules()), 'Probe requires eval mode')
    require(len(indices) > 0 and batch_size > 0 and probe_batch_size > 0, 'Invalid probe scope')
    state_before = training.state_digest(model.state_dict())
    gradients_before = {n: None if p.grad is None else array_digest(p.grad.cpu().numpy())
                        for n, p in model.named_parameters()}
    try:
        print('Replaying saved P1 ON validation before permitting OFF', flush=True)
        metrics, baseline = training.evaluate(model, inputs, 'P1', indices, batch_size, device, scaler)
        replay = compare_replay(baseline, saved_arrays)
        replay['mae_abs_difference'] = abs(metrics['all_nodes']['mae_macro']-saved_metrics['all_nodes']['mae_macro'])
        require(replay['mae_abs_difference'] <= REPLAY_MAE_ATOL, 'ON metric replay failed; OFF pass blocked')
        write_json(output_dir/'replay.json', replay)
        # No output detail tensors are retained across batches.
        on_predictions, predictions, associations, counts, matched_counts, ages = [], [], [], [], [], []
        totals = PathwayTotals()
        replay['paired_on_prediction_abs_max_vs_original_batch'] = 0.
        for start in range(0, len(indices), probe_batch_size):
            selected = indices[start:start+probe_batch_size]
            batch = inputs.batch('val', selected, device, new_reports=True)
            batch_before = tensor_tree_digest(batch)
            with torch.inference_mode():
                on = forward(model, batch, scaler, True)
                actual_on = on['prediction'].cpu().numpy()
                original_on = baseline['prediction'][start:start+len(selected)]
                replay['paired_on_prediction_abs_max_vs_original_batch'] = max(
                    replay['paired_on_prediction_abs_max_vs_original_batch'],
                    float(np.abs(actual_on.astype(np.float64)-original_on).max()))
                require(np.allclose(actual_on, original_on,
                                    atol=REPLAY_ATOL, rtol=REPLAY_RTOL), 'Paired ON differs from verified replay')
                off = forward(model, batch, scaler, False)
                restored = forward(model, batch, scaler, True)
                verify_intervention(on, off, restored, model.branch.graph)
                del restored
                totals.update(on, off)
                on_predictions.append(on['prediction'].cpu().numpy())
                predictions.append(off['prediction'].cpu().numpy())
                associations.append(on['report_association'].cpu().numpy())
                count, matched, age = report_support(batch)
                counts.extend(count.tolist())
                matched_counts.extend(matched.tolist())
                ages.extend(age.tolist())
            require(tensor_tree_digest(batch) == batch_before, 'Probe mutated input batch/native incident inputs')
            del on, off, batch
            print(f'Paired ON/OFF/ON: {min(start+probe_batch_size, len(indices))}/{len(indices)} rows', flush=True)
        # Use actual paired microbatch ON, not a differently rounded baseline.
        return finish_probe(inputs, indices, baseline, on_predictions, predictions, associations, counts, matched_counts,
                            ages, totals, regions, output_dir, replay)
    finally:
        require(training.state_digest(model.state_dict()) == state_before, 'Probe mutated model state')
        require(gradients_before == {n: None if p.grad is None else array_digest(p.grad.cpu().numpy())
                                   for n, p in model.named_parameters()}, 'Probe changed gradients')


def finish_probe(inputs, indices, baseline, on_predictions, predictions, associations, counts, matched_counts,
                 ages, totals, regions, output_dir, replay):
    off = np.concatenate(predictions)
    on = np.concatenate(on_predictions)
    association = np.concatenate(associations)
    nodes = on.shape[2]
    graph = inputs.network.structure
    edges = np.array([[r['source'], r['destination']] for r in graph['edges']], dtype=np.int64).T
    dynamic = input_regions(association, edges, nodes)
    ages = np.asarray(ages)
    for name, low, high in (('matched_report_age_0_5', 0, 5), ('matched_report_age_5_15', 5, 15),
                            ('matched_report_age_15_60', 15, 61)):
        dynamic[name] = np.broadcast_to(((ages >= low) & (ages < high))[:, None], (len(indices), nodes))
    events = [inputs.events['val'][i] for i in indices]
    cutoff_counts = Counter(row['t0'] for row in events)
    weights = np.array([1./cutoff_counts[row['t0']] for row in events])
    kwargs = dict(on=on, off=off, target=baseline['target'], valid=baseline['valid'])
    masks = dict(regions, **dynamic)
    metrics = {name: paired_metric(**kwargs, mask=mask) for name, mask in masks.items()}
    original_mae = paired_metric(baseline['prediction'], baseline['prediction'], baseline['target'],
                                baseline['valid'], regions['all_nodes'])['on_mae_macro']
    replay['paired_on_mae_abs_difference_vs_original_batch'] = abs(metrics['all_nodes']['on_mae_macro']-original_mae)
    require(replay['paired_on_mae_abs_difference_vs_original_batch'] <= REPLAY_MAE_ATOL,
            'Paired ON metric differs from original batch replay')
    write_json(output_dir/'replay.json', replay)
    sensitivity = {name: paired_metric(**kwargs, mask=mask, weights=weights) for name, mask in masks.items()}
    sample_rows = []
    potential = dynamic['report_associated_nodes'] | dynamic['potential_propagation_only_nodes']
    for j, row in enumerate(events):
        metric = paired_metric(on[j:j+1], off[j:j+1], baseline['target'][j:j+1], baseline['valid'][j:j+1],
                               np.ones(nodes, bool))
        sample_rows.append(dict(sample_index=int(row['sample_index']), incident_id=row['incident_id'], t0=row['t0'],
            report_entities=counts[j], matched_report_entities=matched_counts[j],
            associated_edges=int((association[j] > 0).sum()),
            associated_nodes=int(dynamic['report_associated_nodes'][j].sum()),
            potential_component_nodes=int(potential[j].sum()),
            youngest_matched_recorded_report_age_minutes=float(ages[j]) if np.isfinite(ages[j]) else None,
            unique_cutoff_sensitivity_weight=float(weights[j]),
            on_mae=None if metric is None else metric['on_mae_macro'],
            off_mae=None if metric is None else metric['off_mae_macro'],
            off_minus_on_mae=None if metric is None else metric['off_minus_on_mae']))
    write_rows(output_dir/'samples.csv', sample_rows)
    pathway, curves = totals.result(len(indices))
    windows = np.stack((np.arange(5., 65., 5.), np.arange(10., 70., 5.)), -1)
    np.savez_compressed(output_dir/'paired_predictions.npz', on_prediction=on, off_prediction=off,
        target=baseline['target'], valid=baseline['valid'], sample_indices=baseline['sample_indices'],
        station_ids=np.asarray(graph['station_ids']), report_association=association,
        direct_report_node_mask=dynamic['report_associated_nodes'], potential_component_mask=potential,
        unique_cutoff_weights=weights, target_windows_minutes=windows)
    np.savez_compressed(output_dir/'pathway_curves.npz', **curves, edge_index=edges,
        station_ids=np.asarray(graph['station_ids']), target_windows_minutes=windows,
        substep_start_minutes=np.arange(curves['capacity_on_mean'].shape[0])*1.25,
        state_minutes=np.arange(curves['states_on_mean'].shape[0])*1.25)
    supported = association > 0
    internal_edges = (edges >= 0).all(0)
    return dict(status='P1_REPORT_CAPACITY_PROBE_PASS', validation_rows=len(indices),
        distinct_cutoffs=len(cutoff_counts), replay=replay, metrics=metrics,
        unique_cutoff_sensitivity_metrics=sensitivity, pathway=pathway,
        coverage=dict(rows_with_matched_reports=int(supported.any(1).sum()),
                      rows_with_matched_reports_fraction=float(supported.any(1).mean()),
                      matched_report_row_occurrences=sum(matched_counts), report_row_occurrences=sum(counts),
                      supported_sample_edge_positions=int(supported.sum()), sample_edge_positions=int(supported.size),
                      internal_edges=int(internal_edges.sum()),
                      sample_internal_edge_positions=int(len(indices)*internal_edges.sum()),
                      edges_ever_supported=int(supported.any(0).sum()),
                      directly_associated_sample_node_positions=int(dynamic['report_associated_nodes'].sum()),
                      sample_node_positions=len(indices)*nodes),
        difference_sign='OFF_MAE_minus_ON_MAE; positive_means_ON_better_on_identical_observations',
        report_age_semantics='youngest_matched_recorded_report_age_at_cutoff; not_actual_incident_onset',
        potential_region_semantics='weak_component_envelope_of_report_edge_endpoints; not_observed_impact',
        cutoff_weighting='each_row_1_over_number_of_evaluated_rows_with_same_t0; sensitivity_only',
        statistical_independence_claim=False, predictive_gain_claim=False, causal_claim=False,
        true_physical_capacity_claim=False, online_semantics_certified=False, test_accessed=False,
        optimizer_updates=0, model_state_preserved=True, final_on_restored=True,
        original_IGSTGNN_incident_inputs_preserved=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--data-dir', type=Path, default=REPO.parent/'data/chronological/Contra_Costa_v8_dev')
    p.add_argument('--history-dir', type=Path)
    p.add_argument('--sensors', type=Path, default=REPO.parent/'data/xtraffic/Contra_Costa/sensors.csv')
    p.add_argument('--network-dir', type=Path)
    p.add_argument('--report-bundle', type=Path, default=Path(__file__).with_name('report_metadata_m42'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--probe-batch-size', type=int, default=8)
    args = p.parse_args(argv)
    require(args.probe_batch_size > 0, 'Probe batch size must be positive')
    root = args.run_dir.resolve() if args.run_dir else discover_run(REPO/'experiments/chronological_runs')
    require(not (root/'capacity_continuation.lock').exists(), 'Training continuation is active; probe a stopped run')
    identity = read_json(root/'identity.json')
    before = snapshot_run(root)
    session = (args.output_dir or root/('report_capacity_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S')
                                      +'_'+uuid.uuid4().hex[:6])).resolve()
    history = args.history_dir or discover_history([args.data_dir, args.data_dir.parent,
        REPO/'experiments/chronological_runs', REPO.parent/'论文学习/研究开发_20260911'],
        identity['inputs_sha256']['data/summary.json'])
    require(history is not None, 'Original history not found; set --history-dir')
    directories = {k: v.resolve() for k, v in dict(data=args.data_dir, history=history,
        network=args.network_dir or root/'network_pack', metadata=Path(__file__).with_name('physics_metadata'),
        reports=args.report_bundle).items()}
    require(session != root and not any(session.is_relative_to(d) for d in list(directories.values())
                                        +[root/arm for arm in training.ARMS]), 'Output overlaps protected inputs/arms')
    session.mkdir(parents=True, exist_ok=False)
    print('Run directory: '+str(root)+'\nProbe directory: '+str(session), flush=True)
    original = inputs = None
    try:
        verify_sources(identity)
        verify_inputs(identity, directories, args.sensors.resolve())
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(3)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
        write_json(session/'invocation.json', dict(run_dir=str(root), directories={k: str(v) for k, v in directories.items()},
            sensors=str(args.sensors.resolve()), origin_identity=identity, runtime=runtime_environment(device),
            replay_batch_size=identity['batch_size'], probe_batch_size=args.probe_batch_size,
            script_sha256=sha256(Path(__file__)), protected_artifacts_before=before,
            probe_dependency_sha256={name: sha256(REPO/name) for name in (
                'experiments/chronological/continue_incident_capacity.py',
                'experiments/chronological/resume_incident_capacity_checked.py')},
            inference_environment_may_differ=True, replay_tolerances=dict(atol=REPLAY_ATOL, rtol=REPLAY_RTOL,
                                                                         mae_atol=REPLAY_MAE_ATOL),
            test_accessed=False, optimizer_updates=0))
        original = OriginalCapacityInputs(directories['data'], directories['history'], args.sensors,
            directories['metadata'], Path(__file__).with_name('incident_corridor_selection_v1.json'))
        inputs = CapacityDevelopmentInputs(original, directories['network'], directories['history'],
                                          directories['reports'], ramp_exchanges=identity['ramp_exchanges'])
        require(inputs.fingerprints == identity['inputs_sha256'], 'Reconstructed input fingerprints differ')
        protocol = read_json(PROTOCOL)
        require(len(inputs.events['val']) == protocol['val_samples'] and len(original.events) == protocol['train_samples']
                and len(original.station_ids) == protocol['stations'], 'Frozen task shape changed')
        require(identity['batch_size'] == (2 if identity['check'] else protocol['batch_size'])
                and identity['sample_scope'] == ('first_4_train_val' if identity['check'] else 'original_full_train_val'),
                'Frozen sample scope changed')
        indices = list(range(min(4, len(inputs.events['val'])) if identity['check'] else len(inputs.events['val'])))
        regions = regions_from_pack(directories['network'], identity, original.station_ids)
        with np.load(root/'P1/best_validation_predictions.npz', allow_pickle=False) as stored:
            saved = {key: stored[key] for key in stored.files}
        expected_target, expected_valid = inputs.targets('val', indices)
        expected_arrays = dict(target=array_digest(expected_target.numpy()), valid=array_digest(expected_valid.numpy()),
            sample_indices=array_digest(np.array([int(inputs.events['val'][i]['sample_index']) for i in indices])))
        audited = audit_arm(root/'P1', 'P1', identity, protocol, regions, expected_arrays, original.scaler['std'])
        write_json(session/'checkpoint_audit.json', audited)
        set_seed(identity['seed'])
        backbone = make_model(directories['data'], len(original.station_ids), device, 'fixed')
        require(training.state_digest(backbone.state_dict()) == identity['common_backbone_sha256'], 'Backbone initialization changed')
        graph, weights = inputs.network.graph(device)
        set_seed(identity['seed']+4200)
        branch = IncidentCapacityBranch(graph, weights, forecast_dim=backbone._forecast_dim).to(device)
        require(training.state_digest(branch.state_dict()) == identity['common_branch_sha256'], 'Branch initialization changed')
        model = CapacityAugmentedIGSTGNN(backbone, branch).to(device)
        best = torch.load(root/'P1/best_model.pt', map_location='cpu', weights_only=False)
        require(best['identity'] == dict(identity, arm='P1') and best['epoch'] == audited['best_epoch'],
                'Best checkpoint changed during audit')
        for name, value in branch.named_buffers():
            require(torch.equal(value.cpu(), best['model_state']['branch.'+name]), 'Frozen branch buffer changed: '+name)
        model.load_state_dict(best['model_state'], strict=True)
        del best, expected_target, expected_valid
        model.eval()
        report = run_probe(model, inputs, indices, identity['batch_size'], args.probe_batch_size, device,
                           original.scaler, saved, read_json(root/'P1/best_validation_metrics.json'), regions, session)
        verify_sources(identity)
        verify_inputs(identity, directories, args.sensors.resolve())
        require(snapshot_run(root) == before, 'Source run artifacts changed during probe')
        report.update(run_dir=str(root), arm='P1', best_epoch=audited['best_epoch'],
                      completed_epoch=audited['completed_epoch'], check_subset=identity['check'],
                      scientific_status='ENGINEERING_CHECK_ONLY' if identity['check'] else 'EXPLORATORY_CHECKPOINT_SENSITIVITY',
                      checkpoint_sha256=before['P1/best_model.pt'], protected_artifacts_unchanged=True,
                      runtime=runtime_environment(device), saved_origin_environment=identity['environment'])
        report['outputs_sha256'] = {p.name: sha256(p) for p in session.iterdir() if p.is_file()}
        write_json(session/'report.json', report)
        brief = {name: None if report['metrics'][name] is None else {k: report['metrics'][name][k] for k in
                 ('on_mae_macro', 'off_mae_macro', 'off_minus_on_mae', 'valid_count', 'prediction_abs_change_max')}
                 for name in ('all_nodes', 'report_associated_nodes', 'potential_propagation_only_nodes',
                              'no_report_supported_samples')}
        print(json.dumps(dict(status=report['status'], best_epoch=report['best_epoch'],
            completed_epoch=report['completed_epoch'], check_subset=identity['check'], regions=brief,
            per_horizon_off_minus_on=report['metrics']['all_nodes']['per_horizon_off_minus_on'],
            pathway=report['pathway'], coverage=report['coverage'], report=str(session/'report.json')),
            indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        write_json(session/'failure.json', dict(status='P1_REPORT_PROBE_FAILED', error_type=type(error).__name__,
            error=str(error), protected_artifacts_unchanged=snapshot_run(root) == before, test_accessed=False,
            optimizer_updates=0, completion_claim=False))
        raise
    finally:
        if inputs is not None:
            inputs.close()
        if original is not None:
            original.close()


if __name__ == '__main__':
    main()
