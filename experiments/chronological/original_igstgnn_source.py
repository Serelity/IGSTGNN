"""Isolated, hash-checked author source and safe checkpoint/log readers.

The frozen files are byte copies of Git object 558ab01. This identifies code,
not the historical server execution. No live ``src`` module is imported here.
Data access belongs to original_igstgnn_runtime. The original dataloader and
engine are exposed unchanged for their parser/batch/metric semantics; their
legacy load_dataset/train methods include test access and are not audit entry
points. The new runners explicitly construct train/val and omit final test.
"""
import ast
import builtins
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import ModuleType, SimpleNamespace
import zipfile

import torch


VENDOR = Path(__file__).resolve().parent / 'original_igstgnn_558ab01'
MANIFEST_SHA256 = 'fdaf7bd006f70e86d6a1ee3069f858387a7c541149adba55a96f82b1c7879531'
SOURCE_HASHES = {
    'model.py': '18d07e0e4a33b061ec5a69eb1f102dc5120c7cd9275b39a2637d34c43faf288b',
    'dataloader.py': '1c350c7bf266d4bc64fbaaa4fb9bf519792b29e386b6f5213a9b7fcc53bc18eb',
    'metrics.py': '6df3633f8f3007795116a34b2c6d494010e4a2ee6192f5b22ffd6e47b51127fe',
    'graph_algo.py': '49d876dac914eca4f9e75aa2fbc3903c28da8570fd50a5940ef16277cd53af19',
    'base_model.py': '80115d8022303a543c8659a535c41a9ddadfdb5c2a45fe1dc28bafef5075eb83',
    'base_engine.py': 'd6307e55366449cc4ae93abbf08930ed7f606cb2b241828a1da20c683fd11309',
    'engine.py': '330448c8d19836102e1cd59f4bd893f0ae989da0ae63ce1d6abfce5d4e272a77',
    'args.py': '22b3011b652c7b14bfb499f6db6980ec580cf4fd45a996ce1bcedd6a9a8ee171',
    'main_reference.py': '84b5c4fd210eaca52ba39dc648a7db34a307e3ba3fc9b4e00e4ed51e9bae6b26',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_original_source():
    raw = (VENDOR / 'manifest.json').read_bytes()
    require(sha256_bytes(raw) == MANIFEST_SHA256, 'Frozen original source manifest changed')
    manifest = json.loads(raw)
    require(set(manifest['files']) == set(SOURCE_HASHES), 'Frozen original source file set changed')
    for name, expected in SOURCE_HASHES.items():
        value = (VENDOR / name).read_bytes()
        entry = manifest['files'][name]
        require(sha256_bytes(value) == expected == entry['sha256'] and len(value) == entry['bytes'],
                f'Frozen original source changed: {name}')
    return manifest


@lru_cache(maxsize=1)
def _load_original_modules():
    manifest = verify_original_source()
    modules = {}
    aliases = {}

    def original_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in aliases:
            require(level == 0 and bool(fromlist), f'Unexpected frozen source import: {name}')
            return aliases[name]
        require(not name.startswith('src'), f'Unfrozen src dependency: {name}')
        return builtins.__import__(name, globals, locals, fromlist, level)

    order = ('base_model', 'metrics', 'dataloader', 'graph_algo', 'args', 'base_engine', 'engine', 'model')
    for name in order:
        path = VENDOR / f'{name}.py'
        module = ModuleType(f'_igstgnn_original_558ab01.{name}')
        module.__file__ = str(path)
        module.__package__ = '_igstgnn_original_558ab01'
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=original_import)
        # The import shim is local to this module; sys.modules['src.*'] is never
        # rebound, and frozen file bytes are not rewritten to change imports.
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        modules[name] = module
        alias = {'base_model': 'src.base.model', 'metrics': 'src.utils.metrics',
                 'base_engine': 'src.base.engine'}.get(name)
        if alias:
            aliases[alias] = module
    return SimpleNamespace(model=modules['model'], loader=modules['dataloader'],
                           graph=modules['graph_algo'], metrics=modules['metrics'],
                           base_model=modules['base_model'], base_engine=modules['base_engine'],
                           engine=modules['engine'], args=modules['args'], manifest=manifest,
                           source_fingerprint=MANIFEST_SHA256)


def get_original_modules():
    """Return frozen modules, checking on-disk identity even after first import."""
    verify_original_source()
    return _load_original_modules()


def parse_config(log_text):
    """Parse exactly one logged argparse Namespace using literal AST values."""
    lines = [line[line.index('Namespace('):].strip()
             for line in log_text.splitlines() if 'Namespace(' in line]
    require(len(lines) == 1, 'Expected exactly one training Namespace in log')
    try:
        call = ast.parse(lines[0], mode='eval').body
        require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'Namespace' and not call.args, 'Invalid training Namespace')
        config = {}
        for item in call.keywords:
            require(item.arg is not None and item.arg not in config, 'Duplicate or expanded Namespace key')
            value = ast.literal_eval(item.value)
            require(type(value) in (str, int, float, bool, type(None)), 'Non-scalar Namespace value')
            require(not isinstance(value, float) or math.isfinite(value), 'Non-finite Namespace value')
            config[item.arg] = value
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f'Unsafe or invalid training Namespace: {exc}') from exc
    require('implementation' not in config and 'paper_aligned' not in log_text.lower(),
            'paper_aligned identity is not original IGSTGNN')
    required = ('dataset', 'seed', 'bs', 'seq_len', 'horizon', 'input_dim', 'output_dim',
                'incident', 'use_sensor_info', 'num_feat', 'num_hidden', 'node_hidden',
                'time_emb_dim', 'layer', 'k_t', 'k_s', 'gap', 'tpd', 'dropout',
                'lambda_incident', 'sigma_t')
    require(all(key in config for key in required), 'Original training configuration is incomplete')
    require(config.get('model_name', 'igstgnn') == config.get('module_name', 'igstgnn') == 'igstgnn',
            'Training log is not an original IGSTGNN model')
    require(config['incident'] is True and config['use_sensor_info'] is True,
            'Original full-model audit requires incident and sensor inputs enabled')
    for key in ('seed', 'bs', 'seq_len', 'horizon', 'input_dim', 'output_dim', 'num_feat',
                'num_hidden', 'node_hidden', 'time_emb_dim', 'layer', 'k_t', 'k_s', 'gap', 'tpd'):
        require(type(config[key]) is int and config[key] > 0, f'Invalid original config: {key}')
    require(config['seq_len'] == config['horizon'] == 12 and config['input_dim'] == 3
            and config['output_dim'] == config['num_feat'] == 1,
            'Unsupported original history/target/input configuration')
    return config


def parse_training_log(log_text):
    config = parse_config(log_text)
    epoch_pattern = re.compile(r'Epoch:\s*(\d+),.*?Valid Loss:\s*([0-9.eE+-]+),\s*Valid RMSE:\s*([0-9.eE+-]+),\s*Valid MAPE:\s*([0-9.eE+-]+)')
    rows, last_epoch, decrease_epochs = [], None, []
    for line in log_text.splitlines():
        match = epoch_pattern.search(line)
        if match:
            last_epoch = int(match[1])
            rows.append(dict(epoch=last_epoch, validation_mae=float(match[2]),
                             validation_rmse=float(match[3]), validation_mape=float(match[4]),
                             validation_mae_literal=match[2]))
        if 'Val loss decrease from ' in line:
            require(last_epoch is not None, 'Best-model update precedes epoch log')
            decrease_epochs.append(last_epoch)
    require(rows and len({row['epoch'] for row in rows}) == len(rows)
            and [r['epoch'] for r in rows] == list(range(1, len(rows) + 1)),
            'Training epoch log is empty, repeated, or incomplete')
    require(all(math.isfinite(row[key]) for row in rows
                for key in ('validation_mae', 'validation_rmse', 'validation_mape')),
            'Non-finite validation metric in training log')
    require(decrease_epochs, 'Training log has no best-model save evidence')
    best_epoch = decrease_epochs[-1]
    best_row = next(row for row in rows if row['epoch'] == best_epoch)
    stops = re.findall(r'Early stop at epoch\s+(\d+),\s*loss\s*=\s*([0-9.eE+-]+)', log_text)
    require(len(stops) <= 1, 'Ambiguous early-stop records')
    best_literal = stops[0][1] if stops else best_row['validation_mae_literal']
    if stops:
        require(int(stops[0][0]) == rows[-1]['epoch'], 'Early-stop epoch disagrees with completed epochs')
    require(abs(float(best_literal) - best_row['validation_mae']) <= 5.1e-5,
            'Best validation literal disagrees with saved epoch')
    require(best_row['validation_mae'] <= min(row['validation_mae'] for row in rows) + 5.1e-5,
            'Best save record disagrees with validation trajectory')
    counts = {}
    for split in ('train', 'val'):
        found = re.findall(r'\b' + split + r': samples=(\d+), x=\((\d+),\s*(\d+),\s*(\d+)\)', log_text)
        require(len(found) == 1, f'Expected one {split} sample/axis record')
        size, seq, nodes, features = map(int, found[0])
        require(size > 0 and seq == config['seq_len'] and features == config['input_dim'],
                f'Invalid {split} logged axis')
        counts[split] = size
        require('node_num' not in counts or counts['node_num'] == nodes, 'Logged station axes disagree')
        counts['node_num'] = nodes
    parameters = re.findall(r'The number of parameters:\s*(\d+)', log_text)
    require(len(parameters) == 1, 'Expected one parameter-count record')
    return dict(config=config, best_epoch=best_epoch, best_validation_mae=float(best_literal),
                best_validation_mae_literal=best_literal,
                best_validation_mae_precision_digits=len(best_literal.partition('.')[2]),
                best_epoch_metrics=best_row, completed_epochs=rows[-1]['epoch'],
                parameter_count=int(parameters[0]), counts=counts,
                training_rows=rows, log_best_value_is_rounded=True)


@dataclass
class OriginalArchive:
    state_dict: Mapping
    config: dict
    metadata: dict
    provenance: dict
    log_text: str

    @property
    def best_epoch(self):
        return self.metadata['best_epoch']

    @property
    def best_validation_mae(self):
        return self.metadata['best_validation_mae']


def _validate_state_dict(state, config, node_num):
    require(isinstance(state, Mapping) and state and all(isinstance(k, str) for k in state),
            'Checkpoint must be a nonempty plain state_dict mapping')
    require(all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.layout == torch.strided
                for v in state.values()), 'Checkpoint state_dict must contain only dense CPU tensors')
    require(all(not v.is_floating_point() or (v.dtype == torch.float32 and bool(torch.isfinite(v).all()))
                for v in state.values()), 'Expected finite original FP32 checkpoint tensors')
    required_shapes = {'node_emb_u': (node_num, config['node_hidden']),
                       'node_emb_d': (node_num, config['node_hidden']),
                       'embedding.weight': (config['num_hidden'], config['num_feat']),
                       'T_i_D_emb': (config['tpd'], config['time_emb_dim']),
                       'D_i_W_emb': (7, config['time_emb_dim']),
                       'icsf_module.distance_encoder.0.weight': (config['num_hidden'], 3),
                       'icsf_module.icsf_fusion_mlp.3.weight': (config['num_hidden'], config['num_hidden'] * 2),
                       'tiid_module.context_projection.weight': (256, config['num_hidden'])}
    require(not any('context_mlp' in k or 'incident_exchange' in k or 'forecast_fusion' in k
                    for k in state), 'Checkpoint is aligned/modified, not original IGSTGNN')
    for key, shape in required_shapes.items():
        require(key in state and tuple(state[key].shape) == shape, f'Original checkpoint identity mismatch: {key}')
    require('tiid_module.sigma_t' in state and state['tiid_module.sigma_t'].numel() == 1
            and math.isclose(float(state['tiid_module.sigma_t']), config['sigma_t'], rel_tol=1e-6),
            'Checkpoint TIID sigma disagrees with log')
    layers = {int(m[1]) for k in state if (m := re.match(r'layers\.(\d+)\.', k))}
    require(layers == set(range(config['layer'])), 'Checkpoint layer identity mismatch')
    return OrderedDict(state)


def load_archive(archive=None, *, checkpoint=None, train_log=None, model=None):
    """Read one ZIP or one checkpoint+log pair; never extract ZIP members.

    ``model`` is optional. When supplied the complete mapping is checked by
    load_state_dict(strict=True), beyond the architecture identity sentinels.
    Unsafe pickle fallback is deliberately unavailable.
    """
    require((archive is not None) != (checkpoint is not None or train_log is not None),
            'Provide exactly one archive or a checkpoint plus training log')
    if archive is not None:
        path = Path(archive).resolve()
        require(path.is_file() and 'paper_aligned' not in path.name.lower(), 'Invalid original ZIP identity')
        with zipfile.ZipFile(path) as zipped:
            files = [item for item in zipped.infolist() if not item.is_dir()]
            weights = [item for item in files if re.fullmatch(r'final_model_s\d+\.pt', Path(item.filename).name)]
            logs = [item for item in files if re.fullmatch(r'record_igstgnn_s\d+_[^/]+\.log', Path(item.filename).name)]
            require(len(weights) == len(logs) == 1 and len({i.filename for i in files}) == len(files),
                    'ZIP must contain exactly one original checkpoint and training log')
            require(weights[0].file_size <= 1 << 30 and logs[0].file_size <= 16 << 20,
                    'Archive member exceeds supported size')
            weight_bytes, log_bytes = zipped.read(weights[0]), zipped.read(logs[0])
            provenance = dict(kind='zip', path=str(path), sha256=file_sha256(path),
                              checkpoint_member=weights[0].filename, log_member=logs[0].filename,
                              zip_extracted=False)
    else:
        require(checkpoint is not None and train_log is not None, 'Checkpoint needs its original training log')
        weight_path, log_path = Path(checkpoint).resolve(), Path(train_log).resolve()
        require(weight_path.is_file() and log_path.is_file(), 'Original checkpoint or log is missing')
        weight_bytes, log_bytes = weight_path.read_bytes(), log_path.read_bytes()
        provenance = dict(kind='checkpoint_and_log', path=str(weight_path), log_path=str(log_path),
                          zip_extracted=False)
    log_text = log_bytes.decode('utf-8-sig')
    metadata = parse_training_log(log_text)
    config = metadata['config']
    if archive is not None:
        for member in (provenance['checkpoint_member'], provenance['log_member']):
            match = re.search(r'_s(\d+)(?:[_.])', Path(member).name)
            require(match is not None and int(match[1]) == config['seed'],
                    'Archive member seed disagrees with training Namespace')
    state = torch.load(io.BytesIO(weight_bytes), map_location='cpu', weights_only=True)
    state = _validate_state_dict(state, config, metadata['counts']['node_num'])
    if model is not None:
        model.load_state_dict(state, strict=True)
        require(model.param_num() == metadata['parameter_count'], 'Checkpoint/log parameter count mismatch')
    provenance.update(checkpoint_sha256=sha256_bytes(weight_bytes), training_log_sha256=sha256_bytes(log_bytes),
                      weights_only=True, checkpoint_format='plain_state_dict',
                      code_reference='558ab01', historical_execution_commit_independently_certified=False,
                      full_strict_model_load_checked=model is not None)
    return OriginalArchive(state, config, metadata, provenance, log_text)
