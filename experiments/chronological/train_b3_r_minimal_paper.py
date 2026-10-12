"""Train B3R_MIN on the unchanged, fingerprinted author data release.

Uses the existing published-data engine for batches, masking, curriculum and
validation metrics. A different identity prevents resuming chronological runs.
"""
import argparse
import copy
from datetime import datetime
import gc
import logging
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import set_seed, sha256
from experiments.chronological.train import cpu_state, save_checkpoint
from src.engines.igstgnn_engine import IGSTGNN_Engine
from src.models.igstgnn import IGSTGNN
from src.models.incident_spatial import configure_spatial_mode, require
from src.models.incident_spatial_residual import from_canonical_b3
from src.models.incident_spatial_minimal import from_residual
from src.utils.graph_algo import normalize_adj_mx
from src.utils.igstgnn_published import PublishedData
from src.utils.metrics import compute_all_metrics, masked_mae

HERE = Path(__file__).resolve().parent
ARM = 'B3R_MIN'
SCHEMA = 'b3r_minimal_published_run_v1'
PROTOCOL = HERE/'b3_r_minimal_paper_v1.json'


def default_data():
    if os.environ.get('IGSTGNN_PAPER_DATA_DIR'):
        return Path(os.environ['IGSTGNN_PAPER_DATA_DIR'])
    candidates = [REPO/'data/xtraffic/Contra_Costa', REPO/'../data/xtraffic/Contra_Costa']
    return next((path for path in candidates if (path/'incident_all.npy').is_file()), candidates[0])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=default_data())
    parser.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--epochs', type=int, help='Absolute target epoch; default 10, or 1 with --check')
    parser.add_argument('--batch-size', type=int, help='Override training/validation batch size; default protocol value (48); --check always uses 2')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--check', action='store_true', help='First four train/val rows; engineering only')
    parser.add_argument('--evaluate-test', action='store_true', help='Explicitly evaluate saved best weights on test')
    args = parser.parse_args(argv)
    args.epochs = (1 if args.check else 10) if args.epochs is None else args.epochs
    require(1 <= args.epochs <= 100 and args.threads >= 1 and 0 <= args.seed < 2**32,
            'Invalid epochs/threads/seed')
    require(args.batch_size is None or args.batch_size >= 1, 'Batch size must be a positive integer')
    require(not args.check or (args.epochs == 1 and not args.evaluate_test),
            'Engineering check is one small epoch without test evaluation')
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_b3r_min_paper_'+
                       ('check_' if args.check else '')+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'graph_dir', 'protocol', 'run_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def load_protocol(args):
    protocol = legacy.read_json(args.protocol)
    require(protocol == legacy.read_json(PROTOCOL), 'Published protocol changed; use a separate version')
    # Keep release/data/model settings frozen. The effective training batch is
    # recorded in both the protocol and run identity, so incompatible resumes fail.
    if args.batch_size is not None:
        protocol = dict(protocol, batch_size=args.batch_size)
    return protocol


def verify_release(directory, protocol):
    hashes = {}
    for name, expected in protocol['data_sha256'].items():
        path = directory/name
        require(path.is_file(), f'Author release file missing: {path}; use --data-dir with the original release')
        hashes[name] = sha256(path)
        require(hashes[name] == expected, f'Author release fingerprint mismatch: {name}; v8 is not interchangeable')
    return hashes


def build_model(args, nodes, graph, device):
    set_seed(args.seed)
    adjacency = np.load(args.data_dir/'adj_matrix.npy', allow_pickle=False)
    require(adjacency.shape == (nodes, nodes) and np.isfinite(adjacency).all(), 'Published adjacency axis mismatch')
    supports = [torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
                for a in normalize_adj_mx(adjacency, 'doubletransition')]
    reference = IGSTGNN(
        model_args=dict(num_feat=1, num_hidden=32, node_hidden=12, time_emb_dim=12,
                        layer=5, k_s=2, k_t=3, tpd=288, dropout=.1, gap=3, sigma_t=1.,
                        lambda_incident=1., adjs=supports, incident_schema='report_location_v1',
                        time_response='fixed', incident_routing='none'),
        node_num=nodes, input_dim=3, output_dim=1, seq_len=12, horizon=12,
        dataset='Contra_Costa', data_path=str(args.data_dir), use_sensor_info=False).to(device)
    reference = configure_spatial_mode(reference, 'state_restricted', copy.deepcopy(graph).to(device))
    residual, _ = from_canonical_b3(reference)
    model, certificate = from_residual(residual, clock_offset_slots=0)
    certificate['prediction_inputs'] = ['published_historical_flow', 'published_history_tod_dow',
                                        'published_adjacency', 'candidate_directed_graph']
    return model, certificate


def source_fingerprints():
    names = ('experiments/chronological/train_b3_r_minimal_paper.py', 'src/utils/igstgnn_published.py',
             'data/xtraffic/prepare_splits.py', 'src/utils/dataloader.py', 'src/utils/metrics.py',
             'src/utils/graph_algo.py', 'src/base/engine.py', 'src/base/model.py',
             'src/engines/igstgnn_engine.py', 'src/models/igstgnn.py',
             'experiments/chronological/train_internal_spatial.py', 'src/models/incident_response.py',
             'src/models/incident_spatial.py', 'src/models/incident_spatial_residual.py',
             'src/models/incident_spatial_minimal.py', 'src/models/incident_capacity_exchange.py')
    return {name: sha256(REPO/name) for name in names}


def make_identity(args, protocol, hashes, graph_info):
    return dict(schema=SCHEMA, data_source='author_model_ready_release', protocol=protocol,
                protocol_sha256=sha256(args.protocol), package=hashes, graph=graph_info,
                graph_inputs_sha256={name: sha256(args.graph_dir/name)
                                     for name in ('graph.json', 'stations.csv', 'source_summary.json')},
                source_sha256=source_fingerprints(), seed=args.seed, engineering_check=args.check,
                batch_size=2 if args.check else protocol['batch_size'], threads=args.threads,
                clock_policy='last_published_history_slot_no_extra_gap', incident_inputs_used=False,
                runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                             device_type=torch.device(args.device).type),
                test_evaluation_policy='explicit_only', test_payload_deserialized=True)


def verify_identity(directory, identity):
    path = directory/'identity.json'
    if path.exists():
        require(legacy.read_json(path) == identity,
                'Run identity changed; use a fresh paper-data directory. Chronological checkpoints cannot be resumed')
    else:
        require({p.name for p in directory.iterdir()} <= {'active.lock'},
                'Use a fresh paper-data directory; existing results are preserved')
        legacy.write_json(path, identity)


def make_engine(args, protocol, data, model, logger):
    optimizer = legacy.optimizer_for(model, protocol)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, protocol['lr_milestones'], gamma=protocol['lr_gamma'])
    batches = data.loaders['train_loader'].num_batch
    return IGSTGNN_Engine(
        device=torch.device(args.device), model=model, dataloader=data.loaders, scaler=data.scaler,
        sampler=None, loss_fn=masked_mae, lrate=protocol['learning_rate'], optimizer=optimizer,
        scheduler=scheduler, clip_grad_value=protocol['clip_grad_norm'], max_epochs=protocol['max_epochs'],
        patience=protocol['patience'], log_dir=str(args.run_dir/ARM), logger=logger, seed=args.seed,
        cl_step=protocol['cl_epoch']*batches, warm_step=protocol['warm_epoch']*batches,
        horizon=protocol['horizon'], incident=False)


def summary_for(saved, args, data):
    row = saved['history'][-1]
    return dict(status='ENGINEERING_CHECK_PASS' if args.check else 'PAUSED_AT_EPOCH_BOUNDARY',
                arm=ARM, data_source='author_model_ready_release', completed_epoch=saved['completed_epoch'],
                best_epoch=row['best_epoch'], best_metric=saved['best_metric'],
                selection_metric='published_engine_validation_masked_mae', global_updates=saved['global_updates'],
                curriculum_length=saved['curriculum_length'], parameters=saved['parameters'],
                batch_size=data.loaders['train_loader'].bs,
                incident_inputs_used=False, tiid_module_present=False, D_module_present=False,
                data_audit=data.audit, test_evaluated=False, predictive_gain_claim=False,
                checkpoint='last.pt', best_model_state_key='best_model_state')


def restore(saved, identity, engine, data):
    require(saved.get('identity') == identity and saved.get('arm') == ARM, 'Checkpoint identity mismatch')
    require(saved['completed_epoch'] == len(saved['history']) and saved['completed_epoch'] >= 1,
            'Invalid committed checkpoint epoch/history')
    engine.model.load_state_dict(saved['model_state'], strict=True)
    engine._optimizer.load_state_dict(saved['optimizer_state'])
    engine._lr_scheduler.load_state_dict(saved['scheduler_state'])
    engine._iter_cnt, engine._cl_len = saved['global_updates'], saved['curriculum_length']
    data.loaders['train_loader'].set_order(saved['train_order'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if torch.device(engine._device).type == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def evaluate_test(engine, data, saved, root, logger):
    engine.model.load_state_dict(saved['best_model_state'], strict=True)
    engine.model.eval()
    predictions, labels = [], []
    loader = data.test_loader(engine._dataloader['val_loader'].bs, logger)
    with torch.inference_mode():
        for x, y in loader.get_iterator():
            x, y, incident, sensor = engine._prepare_batch((x, y))
            pred, target = engine._inverse_transform([engine._predict(x, y, incident, sensor), y])
            predictions.append(pred.cpu())
            labels.append(target.cpu())
    pred, target = torch.cat(predictions), torch.cat(labels)
    null = engine._mask_value(target)
    by_horizon = [compute_all_metrics(pred[:, h], target[:, h], null) for h in range(engine._horizon)]
    report = dict(status='PUBLISHED_TEST_EVALUATION_COMPLETE', best_epoch=saved['history'][-1]['best_epoch'],
                  samples=loader.size, mask_value=float(null),
                  by_horizon=[dict(horizon=h+1, mae=m[0], mape=m[1], rmse=m[2]) for h, m in enumerate(by_horizon)],
                  average=dict(zip(('mae', 'mape', 'rmse'), np.mean(by_horizon, axis=0).tolist())))
    legacy.write_json(root/'test_metrics.json', report)
    return report


def train(args, protocol, data, identity, graph, logger):
    root = args.run_dir/ARM
    root.mkdir(exist_ok=True)
    model, certificate = build_model(args, len(data.station_ids), graph, torch.device(args.device))
    legacy.write_json(args.run_dir/'initialization.json', certificate)
    engine = make_engine(args, protocol, data, model, logger)
    path = root/'last.pt'
    saved = None
    history, best, wait = [], float('inf'), 0
    parameters = sum(p.numel() for p in model.parameters())
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        restore(saved, identity, engine, data)
        history, best, wait = saved['history'], saved['best_metric'], saved['wait']
        # The atomic checkpoint is authoritative after an interrupted JSON write.
        legacy.write_json(root/'history.json', history)
        legacy.write_json(root/'summary.json', summary_for(saved, args, data))
    best_state = saved['best_model_state'] if saved else None
    for epoch in range(len(history)+1, args.epochs+1):
        if wait >= protocol['patience']:
            break
        start = time.perf_counter()
        train_mae, train_mape, train_rmse = engine.train_batch()
        val_mae, val_mape, val_rmse = engine.evaluate('val')
        require(np.isfinite([train_mae, train_rmse, val_mae, val_rmse]).all(), 'Non-finite published loss/metrics')
        improved = val_mae < best
        if improved:
            best, wait, best_state = val_mae, 0, cpu_state(model.state_dict())
        else:
            wait += 1
        lr = engine._optimizer.param_groups[0]['lr']
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, train_mae=float(train_mae), train_rmse=float(train_rmse), train_mape=float(train_mape),
                   validation_mae=float(val_mae), validation_rmse=float(val_rmse), validation_mape=float(val_mape),
                   best_metric=best, best_epoch=epoch if improved else history[-1]['best_epoch'],
                   global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
                   learning_rate_used=lr, next_learning_rate=engine._optimizer.param_groups[0]['lr'],
                   seconds=time.perf_counter()-start)
        history.append(row)
        saved = dict(identity=identity, arm=ARM, completed_epoch=epoch, history=history, parameters=parameters,
                     model_state=cpu_state(model.state_dict()), best_model_state=best_state, best_metric=best, wait=wait,
                     optimizer_state=engine._optimizer.state_dict(), scheduler_state=engine._lr_scheduler.state_dict(),
                     global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
                     train_order=data.loaders['train_loader'].order.copy(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [])
        save_checkpoint(path, saved)
        legacy.write_json(root/'history.json', history)
        legacy.write_json(root/'summary.json', summary_for(saved, args, data))
        print(json_dump(dict(arm=ARM, data_source='author_published', **row)), flush=True)
        gc.collect()
    require(saved is not None, 'No committed training checkpoint')
    result = summary_for(saved, args, data)
    if args.evaluate_test:
        result['test_metrics'] = evaluate_test(engine, data, saved, root, logger)
        result['test_evaluated'] = True
    legacy.write_json(root/'summary.json', result)
    legacy.write_json(args.run_dir/'latest_stage.json', result)
    print(json_dump(dict(status=result['status'], completed_epoch=result['completed_epoch'],
                         best_epoch=result['best_epoch'], best_metric=result['best_metric'],
                         output=str(root/'summary.json'))), flush=True)
    return result


def json_dump(value):
    import json
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def main(argv=None):
    args = parse_args(argv)
    # Reject old runs before reading large files or touching their checkpoints.
    if (args.run_dir/'identity.json').is_file():
        require(legacy.read_json(args.run_dir/'identity.json').get('schema') == SCHEMA,
                'Use a fresh paper-data directory; chronological checkpoints cannot be resumed')
    protocol = load_protocol(args)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()),
            'Requested device unavailable')
    logger = logging.getLogger('b3r_min_paper')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    hashes = verify_release(args.data_dir, protocol)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with legacy.run_lock(args.run_dir):
        data = PublishedData(args.data_dir, protocol, logger, check=args.check)
        graph, _, graph_info = legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
        identity = make_identity(args, protocol, hashes, graph_info)
        verify_identity(args.run_dir, identity)
        legacy.write_json(args.run_dir/'data_audit.json', data.audit)
        session = args.run_dir/('environment_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.json')
        legacy.write_json(session, legacy.environment(device))
        print('Data source: author published release; no v8 rebuild or extra gap', flush=True)
        print('Data directory: '+str(args.data_dir), flush=True)
        print('Run directory: '+str(args.run_dir), flush=True)
        print('Batch size: '+str(data.loaders['train_loader'].bs), flush=True)
        train(args, protocol, data, identity, graph, logger)


if __name__ == '__main__':
    main()
