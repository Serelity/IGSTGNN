"""M4.0 integration checks plus original TRAIN-X state interface; no real Y."""
import argparse
import copy
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.smoke import make_model
from src.models.igstgnn import IGSTGNN
from src.models.incident_capacity_exchange import ExchangeGraph
from src.models.incident_capacity_fusion import IncidentCapacityBranch, CapacityAugmentedIGSTGNN
from src.utils.capacity_fusion_inputs import OriginalCapacityInputs
from src.utils.incident_corridor import read_json, require, sha256, write_json, write_rows

SOURCES = ('src/models/incident_capacity_fusion.py', 'src/models/igstgnn.py',
           'src/utils/capacity_fusion_inputs.py', 'tests/test_incident_capacity_fusion.py',
           'tests/test_capacity_fusion_inputs.py', 'experiments/chronological/check_incident_capacity_fusion.py',
           'experiments/chronological/run_incident_capacity_fusion.sh')


def synthetic_check(output, device):
    graph = ExchangeGraph(3, torch.tensor([[-1, 0, 1, 2], [0, 1, 2, -1]], device=device),
                          torch.ones(3, dtype=torch.bool, device=device),
                          torch.tensor([True, False, True], device=device), evidence_scope='synthetic_only')
    branch = IncidentCapacityBranch(graph, torch.tensor([0., 1., 1., 1.], device=device)).to(device)
    native = IGSTGNN(model_args=dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                     layer=2, k_s=1, k_t=2, tpd=288, dropout=0., gap=3, sigma_t=1., lambda_incident=1.,
                     adjs=[torch.eye(3, device=device)]*2, incident_schema='report_location_v1',
                     time_response='fixed', incident_routing='none'), node_num=3, input_dim=3,
                     output_dim=1, seq_len=12, horizon=12, dataset='synthetic', use_sensor_info=False).to(device).eval()
    model = CapacityAugmentedIGSTGNN(copy.deepcopy(native), branch).eval()
    auxiliary = dict(history=torch.rand(2, 12, 3, 3, device=device),
                     valid=torch.ones(2, 12, 3, 3, dtype=torch.bool, device=device),
                     references=torch.ones(3, 3, device=device), labels=torch.arange(-65., -5., 5., device=device))
    auxiliary['reports'] = dict(weights=torch.tensor([[[0., 1., 0., 0.]]]*2, device=device),
                                ages=torch.tensor([[3.], [4.]], device=device),
                                present=torch.ones(2, 1, dtype=torch.bool, device=device),
                                distance=torch.zeros(2, 1, 4, device=device), confidence=torch.ones(2, 1, device=device))
    trigger = dict(distances=torch.tensor([[[0., .7, 0.], [0., 0., 0.], [0., .2, 1.]]]*2, device=device),
                   report_age_minutes=torch.tensor([3., 4.], device=device),
                   forecast_tod=torch.tensor([20, 21], device=device), forecast_dow=torch.tensor([1, 2], device=device))
    x = torch.rand(2, 12, 3, 3, device=device)
    with torch.no_grad():
        initial_difference = float((model(x, incident_data=trigger, capacity_inputs=auxiliary)
                                    -native(x, incident_data=trigger)).abs().max())
    require(initial_difference == 0, 'Zero projection must preserve the full native forecast')
    # CUDA cuDNN GRU backward needs training mode; retain eval backbone so its
    # dropout/BN do not confound this branch-only synthetic integration check.
    branch.train()
    # Binding synthetic capacity is explicit. It is not an incident label or a
    # requirement that every real report produce an effect/gradient.
    with torch.no_grad():
        branch.coefficients[-1].bias.fill_(.8)
    optimizer = torch.optim.Adam(branch.parameters(), lr=.01)
    records = []
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        r = model(x, incident_data=trigger, capacity_inputs=auxiliary, return_details=True)
        main = (r['prediction']-2).square().mean()
        auxiliary_loss = (r['auxiliary_prediction']-1).square().mean()
        # After the first update also verify main-loss-only connectivity.
        if update == 1:
            main.backward(retain_graph=True)
            main_gradient = float(branch.coefficients[-1].weight.grad.abs().sum())
            require(main_gradient > 0, 'Main prediction does not reach the capacity network in a binding regime')
            optimizer.zero_grad(set_to_none=True)
        else:
            main_gradient = None
        (main + .1*auxiliary_loss).backward()
        gradient = float(branch.coefficients[-1].weight.grad.abs().sum())
        require(gradient > 0, 'Auxiliary gradient missing in binding regime')
        records.append(dict(update=update+1, synthetic_loss_NOT_forecast_metric=float(main.detach()),
                            capacity_gradient_l1=gradient, main_only_capacity_gradient_l1=main_gradient))
        optimizer.step()
    checkpoint = output / 'fusion_synthetic_only.pt'
    torch.save(dict(branch=branch.state_dict(), optimizer=optimizer.state_dict()), checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location=device)
    replay = copy.deepcopy(branch)
    replay.load_state_dict(saved['branch'])
    replay.history.gru.flatten_parameters()
    resumed = torch.optim.Adam(replay.parameters(), lr=.01)
    resumed.load_state_dict(saved['optimizer'])
    with torch.no_grad():
        difference = float((replay(**auxiliary)['features']-branch(**auxiliary)['features']).abs().max())
        on, off, restored = branch(**auxiliary), branch(**auxiliary, incident_enabled=False), branch(**auxiliary)
        for key in ('initial', 'boundary_demand', 'history_coefficients'):
            require(torch.equal(on[key], off[key]), 'Report bypass into ' + key)
        require(torch.equal(on['features'], restored['features']), 'Condition restoration failed')
    for b, opt in ((branch, optimizer), (replay, resumed)):
        opt.zero_grad(set_to_none=True)
        (b(**auxiliary)['auxiliary_prediction']-1).square().mean().backward()
        opt.step()
    continuation = max(float((p-q).detach().abs().max()) for p, q in zip(branch.parameters(), replay.parameters()))
    require(difference == continuation == 0, 'Checkpoint/optimizer continuation mismatch')
    ordinary = IncidentCapacityBranch(copy.deepcopy(graph), branch.outgoing_weights, mode='ordinary').to(device)
    p, n = [sum(v.numel() for v in b.parameters()) for b in (branch, ordinary)]
    return dict(status='SYNTHETIC_FUSION_PASS', graph_scope='synthetic_only', initial_forecast_abs_max=initial_difference,
                binding_capacity_gradient_checks=records, synthetic_updates=2,
                checkpoint_feature_abs_max=difference, continuation_parameter_abs_max=continuation,
                branch_parameters=dict(capacity=p, ordinary=n, ordinary_additional=n-p),
                budget_policy='identical shared widths; ordinary has 244 extra active parameters; not exact matching',
                report_path='coefficients -> capacity -> recurrence -> features -> residual',
                L1_local_control_implemented=False)


def original_train_check(args, output, device):
    adapter = OriginalCapacityInputs(args.data_dir, args.history_dir, args.sensors, args.metadata_dir, args.identity)
    try:
        require(len(adapter.events) == 3604 and len(adapter.station_ids) == 496, 'Expected original 3604/496 axes')
        ledger = adapter.qualification_ledger()
        write_rows(output / 'station_qualification.csv', ledger['stations'])
        write_rows(output / 'road_qualification.csv', ledger['roads'])
        write_rows(output / 'candidate_connection_qualification.csv', ledger['candidate_pairs'])
        write_json(output / 'qualification.json', ledger['summary'])
        graph = adapter.qualified_graph(device)
        branch = IncidentCapacityBranch(graph, torch.empty(0, device=device)).to(device).eval()
        start = time.perf_counter()
        available = torch.zeros(496, dtype=torch.bool, device=device)
        minimum, maximum = 1., 0.
        with torch.no_grad():
            for first in range(0, len(adapter.events), args.batch_size):
                end = min(first+args.batch_size, len(adapter.events))
                batch = adapter.batch(np.arange(first, end), device)
                encoded = branch.history(**batch['capacity_inputs'])
                state = encoded['initial']
                require(state.shape == (end-first, 496, 4) and torch.isfinite(state).all()
                        and ((state >= 0) & (state <= 1)).all(), 'Invalid original-X cutoff state')
                available |= encoded['available'].any(0)
                minimum, maximum = min(minimum, float(state.min())), max(maximum, float(state.max()))
                if first == 0 or end == len(adapter.events) or end % 1024 == 0:
                    print(f'Original TRAIN-X cutoff state: {end}/3604, 496 nodes, no targets or updates', flush=True)
        # Qualification-pending nodes have zero direct augmentation. This is an
        # explicit fallback check, never an experiment proving spatial coverage.
        native = make_model(adapter.data_dir, 496, device, 'fixed').eval()
        model = CapacityAugmentedIGSTGNN(native, branch).eval()
        batch = adapter.batch(np.arange(2), device)
        with torch.no_grad():
            original = native(batch['x'], incident_data=batch['incident'])
            result = model(batch['x'], incident_data=batch['incident'], capacity_inputs=batch['capacity_inputs'], return_details=True)
            difference = float((original-result['prediction']).abs().max())
        require(difference == 0 and result['features'].shape == (2, 12, 496, 16), 'Original axis/fallback mismatch')
        require(all(p.grad is None for p in model.parameters()), 'Real data created gradients')
        return dict(status='ORIGINAL_TRAIN_X_STATE_INTERFACE_PASS', samples=3604, nodes=496,
                    nodes_with_available_history=int(available.sum()), original_trigger_path_preserved=True,
                    state_range_UNTRAINED_NOT_density=[minimum, maximum], source_hashes=adapter.fingerprints,
                    normalization='original_unique_train_X_global_channel_mean_as_asinh_reference_not_capacity',
                    qualification=ledger['summary'], qualified_graph_edges=graph.edges,
                    fusion_test_scope='evidence_pending_zero_increment_fallback_only',
                    initial_forecast_abs_max=difference, real_optimizer_updates=0,
                    real_Y_read=False, validation_test_traffic_read=False, seconds=time.perf_counter()-start)
    finally:
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--history-dir', type=Path)
    parser.add_argument('--sensors', type=Path)
    parser.add_argument('--metadata-dir', type=Path, default=REPO / 'experiments/chronological/physics_metadata')
    parser.add_argument('--identity', type=Path, default=REPO / 'experiments/chronological/incident_corridor_selection_v1.json')
    args = parser.parse_args()
    require(args.batch_size > 0, 'Invalid batch size')
    real_paths = (args.data_dir, args.history_dir, args.sensors)
    require(all(v is None for v in real_paths) or all(v is not None for v in real_paths), 'Supply all three real-input paths')
    require(not args.output_dir.exists(), 'Use a fresh output directory')
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    torch.set_num_threads(3)
    torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, False
    args.output_dir.mkdir(parents=True)
    synthetic = synthetic_check(args.output_dir, device)
    real = original_train_check(args, args.output_dir, device) if args.data_dir else None
    result = dict(schema='incident_capacity_fusion_m40_v1', status='M40_ENGINEERING_CHECK_PASS',
                  synthetic=synthetic, original_train_X=real,
                  scientific_status='NO_REAL_PREDICTIVE_GAIN_OR_REAL_NETWORK_MECHANISM_CLAIM',
                  main_training_ready=False, real_optimizer_updates=0,
                  environment=dict(python=platform.python_version(), torch=str(torch.__version__), device=str(device),
                                   gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
                  sources_sha256={p: sha256(REPO / p) for p in SOURCES})
    write_json(args.output_dir / 'summary.json', result)
    print(__import__('json').dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
