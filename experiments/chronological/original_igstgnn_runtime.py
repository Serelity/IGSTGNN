"""Shared original-release data/runtime boundaries; no test loader or optimizer."""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
CONFIG = dict(dataset='Contra_Costa', model_name='igstgnn', seed=2025, bs=48,
    seq_len=12, horizon=12, input_dim=3, output_dim=1, mode='train', max_epochs=100,
    patience=20, incident=True, use_sensor_info=True, num_feat=1, num_hidden=32,
    node_hidden=12, time_emb_dim=12, layer=5, k_t=3, k_s=2, gap=3, cl_epoch=3,
    warm_epoch=30, tpd=288, lrate=.002, wdecay=1e-5, dropout=.1,
    clip_grad_value=5, icsf_dim=64, module_name='igstgnn', lambda_incident=1., sigma_t=1.)
RELEASE = {
    'incident_all.npy': 'bc6c5608457f88864625d683f9a66d17bc0fc31fe5acba366b6033708a099279',
    'incident_stats.npz': '9f6590a3d1df5a0cd59e00717e3157520a4f5b99996976a4e41ddc6fc28b224c',
    'adj_matrix.npy': 'de2a74cc981853ff81dd494d6e7d212b7ae524fb9d6079c91323e56bcb22254f',
    'sensors.csv': '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094',
    'desc_mapping.json': '9a2ff89b2b0057f8617652d305eee79eb3f04686d4bfc58821ed3271ae8a7b71',
    'type_mapping.json': '6a836068bfcbea9868add90535e8efe329d684a074436ef47c5a53c257fd2a9d'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def default_data_dir():
    candidates = [REPO/'data/xtraffic/Contra_Costa', REPO.parent/'data/xtraffic/Contra_Costa']
    return next((p for p in candidates if p.is_dir()), candidates[0])


def configure_runtime(device, threads, seed=2025):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested CPU/CUDA device is unavailable')
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return dict(python_device=str(device), torch=str(torch.__version__), numpy=str(np.__version__),
        cuda=torch.version.cuda, device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        threads=threads, seed=seed, dtype='float32', deterministic_algorithms=True, tf32=False,
        cudnn_benchmark=False)


def load_array(path):
    """Only the trusted published object array; support its NumPy-2 module path."""
    try:
        return np.load(path, allow_pickle=True)
    except ModuleNotFoundError as error:
        if error.name not in ('numpy._core', 'numpy._core.multiarray'):
            raise
        import sys
        import numpy.core.multiarray as multiarray
        sys.modules.setdefault('numpy._core.multiarray', multiarray)
        return np.load(path, allow_pickle=True)


def load_data(directory, config, modules, logger, *, check=False, training=False):
    directory = Path(directory).resolve()
    hashes = {}
    for name, expected in RELEASE.items():
        path = directory/name
        actual = sha256(path)
        if actual != expected:
            raise ValueError('Published release fingerprint mismatch: '+name)
        hashes[name] = actual
    with (directory/'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    station_ids = np.asarray([int(row['station_id']) for row in rows], dtype=np.int64)
    if len(station_ids) != 496 or len(set(station_ids)) != 496:
        raise ValueError('Expected original 496-station axis')
    args = SimpleNamespace(**dict(config, node_num=496))
    sensors = modules.loader._load_sensor_info(str(directory), 496, args, logger)
    with np.load(directory/'incident_stats.npz', allow_pickle=False) as stats:
        mean, std = float(stats['mean']), float(stats['std'])
        if not bool(stats['normalized']) or not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError('Expected unchanged normalized author release')
    scaler = modules.loader.StandardScaler(mean, std)
    wanted = ('train', 'val') if training else ('val',)
    # Authenticate the single published array, then reproduce its original
    # order split. Arbitrary same-shape split files are not authenticated data.
    all_samples = load_array(directory/'incident_all.npy')
    if len(all_samples) != 5585:
        raise ValueError('Expected 5585 published samples')
    loaders, identities = {}, {}
    for split in wanted:
        start, stop = (0, 3909) if split == 'train' else (3909, 4746)
        samples = all_samples[start:stop]
        origin = 'incident_all.npy author-order slice'
        if len(samples) != stop-start:
            raise ValueError('Original '+split+' sample count differs')
        if check:
            samples = samples[:4]
        for sample in samples:
            for key in ('x_data', 'y_data'):
                value = np.asarray(sample[key])
                if value.shape != (12, 496, 3) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError('Invalid published '+split+' '+key)
        batch = 2 if check else int(config['bs'])
        loaders[split+'_loader'] = modules.loader.IncidentDataLoader(samples, True, batch, logger,
            input_dim=3, sensor_info=sensors)
        identities[split] = dict(origin=origin, bounds_half_open=[start, stop],
            selected_indices=list(range(start, start+len(samples))), count=len(samples),
            batch_size=batch, partition=[min(batch, len(samples)-i) for i in range(0, len(samples), batch)])
    roads = {}
    for index, row in enumerate(rows):
        road = str(row.get('Fwy', row.get('Freeway', row.get('Road', 'unknown'))))
        direction = str(row.get('Direction', '')).strip()
        name = road+('-'+direction if direction and not road.upper().endswith('-'+direction.upper()) else '')
        roads.setdefault(name, []).append(index)
    return SimpleNamespace(loaders=loaders, scaler=scaler, sensors=sensors, config=vars(args),
        station_ids=station_ids, roads=roads, directory=directory,
        identity=dict(data_source='author_model_ready_release', hashes=hashes, splits=identities,
            normalization=dict(mean=mean, std=std, refitted=False), sensor_parser='unchanged_558ab01',
            test_file_opened=False, test_payload_deserialized=all_samples is not None, test_evaluated=False,
            absolute_calendar_order_independently_verified=False))


def build_model(data, modules, device):
    graphs = modules.graph.normalize_adj_mx(np.load(data.directory/'adj_matrix.npy', allow_pickle=False), 'doubletransition')
    config = dict(data.config, data_path=str(data.directory), device=str(device),
                  adjs=[torch.as_tensor(np.asarray(g), device=device) for g in graphs])
    model = modules.model.IGSTGNN(model_args=config, node_num=496, input_dim=3,
        output_dim=1, seq_len=12, horizon=12, dataset='Contra_Costa', data_path=str(data.directory),
        use_sensor_info=True)
    return model.to(device)


def device_batch(batch, device):
    x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
    y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
    incident = dict(incident=torch.as_tensor(batch['incident_features'], dtype=torch.float32, device=device),
        position=torch.as_tensor(batch['incident_position'], device=device),
        distances=torch.as_tensor(batch['incident_distances'], dtype=torch.float32, device=device))
    sensors = {key: torch.as_tensor(value, device=device) for key, value in batch['sensor_data'].items()}
    return x, y, incident, sensors


def tensor_tree_hash(value):
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                add(child)
        elif isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(item.tobytes())
        else:
            digest.update(repr(item).encode())
    add(value)
    return digest.hexdigest()


def rng_fingerprint():
    return tensor_tree_hash(dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
        python=random.getstate(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))


def native_metrics(prediction, target, modules):
    p, y = torch.as_tensor(prediction, dtype=torch.float32), torch.as_tensor(target, dtype=torch.float32)
    null = y.min() if y.min() < 1 else y.new_tensor(0.)
    return dict(mae=float(modules.metrics.masked_mae(p, y, null)),
        rmse=float(modules.metrics.masked_rmse(p, y, null)),
        mape_fraction=float(modules.metrics.masked_mape(p, y, null)), mask_value=float(null))


@contextmanager
def run_lock(directory):
    stream = (Path(directory)/'active.lock').open('a+b')
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise ValueError('Another process owns this run directory')
    try:
        yield
    finally:
        stream.close()
