"""v12n frozen inference/export and independently replayable NumPy statistics."""

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import vector_correction_geometry as geometry

PROTOCOL = Path(__file__).with_name('vector_correction_geometry_v12n.json')
PROTOCOL_SHA256 = 'e999d0b9c124aa58c72868fe445651f2a811f6d2b94b6b48ba3ce38e55c29a0a'
POLICY = 'candidate_early_only'
PATH = 'protected_at_protected'
REGIONS = ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all')
require = geometry.require


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def decode(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)


def identical(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ordinary(path):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), f'Symlink path rejected: {path}')
    return path


def relative_file(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == name,
            'Unsafe artifact path')
    return ordinary(root / relative)


def code_hashes(protocol):
    names = list(protocol['source']['producer_code_sha256']) + [
        str(p.relative_to(REPO)) for p in (Path(__file__), Path(geometry.__file__), PROTOCOL,
            PROTOCOL.with_name('run_vector_correction_geometry.sh'))]
    return {name: sha256(REPO / name) for name in names}


def load_protocol():
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, 'Frozen v12n protocol changed')
    protocol = decode(PROTOCOL.read_bytes())
    for name, digest in protocol['source']['producer_code_sha256'].items():
        require(sha256(REPO / name) == digest, f'Original inference code changed: {name}')
    require(tuple(x['name'] for x in protocol['geometry']['categories_in_order']) == geometry.CATEGORIES,
            'Geometry categories differ from frozen design')
    return protocol


class Source:
    """Allowlisted selected-output files, with producer hashes and identity checks."""

    def __init__(self, root, protocol, *, allow_engineering=False):
        self.root, self.protocol = ordinary(root), protocol
        require(self.root.is_dir() and not self.root.name.endswith('.partial'), 'Require a completed v12m source')
        self.hashes, self.records = {}, {}
        self.summary = decode(self.read('summary.json', listed=False))
        s, spec = self.summary, protocol['source']
        self.engineering = s.get('engineering_check')
        require(type(self.engineering) is bool, 'Missing engineering source flag')
        require(not self.engineering or allow_engineering, 'Engineering source rejected for formal audit')
        expected_status = 'ENGINEERING_CHECK_PASS' if self.engineering else spec['required_status']
        require(s.get('status') == expected_status and s.get('protocol_sha256') == spec['protocol_sha256'],
                'Incomplete/wrong source protocol')
        self.frozen = decode((REPO / spec['protocol_file']).read_bytes())
        self.inherited = decode((REPO / spec['inherited_protocol_file']).read_bytes())
        require(identical(s['frozen_protocol'], self.frozen) and identical(s['inherited_protocol'], self.inherited),
                'Source frozen/inherited protocol changed')
        require(s['code_sha256'] == spec['producer_code_sha256'], 'Source producer code identity changed')
        for field in ('all_selectors_frozen_before_audit_evaluation', 'all_output_paths_frozen_before_audit_evaluation',
                      'paired_initialization_exact'):
            require(s.get(field) is True, f'Source boundary changed: {field}')
        require(s.get('unrestricted_at_protected_evaluated') is False and s.get('main_training_ready') is False,
                'Source output/information boundary changed')
        for field, value in self.frozen['information_boundary'].items():
            require(identical(s.get(field), value), f'Source information boundary changed: {field}')
        if not self.engineering:
            require(s['environment']['git_head'] == spec['git_head'], 'Require the frozen formal producer commit')
        self.identity = decode(self.read('run_identity.json'))
        self.plan = decode(self.read('eligibility.json'))
        self.endpoints = decode(self.read('selected_endpoints_frozen.json'))
        self.paths = decode(self.read('evaluation_paths_frozen.json'))
        require(self.endpoints == s['selected_endpoints_frozen'] and self.paths == s['evaluation_paths_frozen'],
                'Frozen manifests disagree with summary')
        training = copy.deepcopy(self.inherited['training'])
        training['objective'] = 'v12m shared candidate_early trajectory; unchanged vector energy penalty'
        self.seeds = self.inherited['check']['seeds'] if self.engineering else protocol['seeds']
        if self.engineering:
            training['epochs'] = self.inherited['check']['epochs']
        self.indices = copy.deepcopy({p: item['indices'] for p, item in self.plan.items()})
        if self.engineering:
            limit = self.inherited['check']['samples_per_cohort_period']
            self.indices = {p: {c: v[:limit] for c, v in cohorts.items()} for p, cohorts in self.indices.items()}
        counts = {p: {c: len(v) for c, v in cohorts.items()} for p, cohorts in self.indices.items()}
        full_counts = {p: {c: len(v) for c, v in item['indices'].items()} for p, item in self.plan.items()}
        require(full_counts == protocol['expected_phase_samples'] and counts == s['phase_samples'], 'Source sample budget changed')
        for key, value in {'inputs': s['inputs'], 'code_sha256': s['code_sha256'],
                'protocol_sha256': s['protocol_sha256'], 'engineering_check': self.engineering,
                'indices': self.indices, 'seeds': self.seeds, 'effective_training': training,
                'output_policies': self.frozen['policies'], 'output_paths': self.frozen['output_paths']}.items():
            require(identical(self.identity.get(key), value), f'Run identity mismatch: {key}')
        require(identical(s['effective_training'], training), 'Source effective training changed')
        fits = len(self.seeds) * len(protocol['arms'])
        require(s['budget'] == {'fits': fits, 'trajectory_epochs': fits * training['epochs'],
                'selected_endpoints': fits * 2, 'output_paths': fits * 3}, 'Incomplete source trajectory budget')
        require(len(self.endpoints) == fits * 2 and len(self.paths) == fits * 3, 'Incomplete frozen manifests')
        require(set(s['runs']) == set(map(str, self.seeds)), 'Source seed set changed')
        self.details = {}
        for seed in self.seeds:
            require(set(s['runs'][str(seed)]) == {f'{arm}_s{seed}' for arm in protocol['arms']}, 'Source arm set changed')
            for arm in protocol['arms']:
                name = f'{arm}_s{seed}'
                detail = decode(self.read(f'{name}/fit_summary.json'))
                declared = s['runs'][str(seed)][name]
                for key in detail:
                    if key != 'policies':
                        require(identical(detail[key], declared.get(key)), f'Fit summary differs: {name}/{key}')
                require(detail['arm'] == arm and type(detail['seed']) is int and detail['seed'] == seed
                        and detail['loss'] == 'candidate_early' and detail['epochs'] == training['epochs'],
                        'Fit identity/budget changed')
                for policy, choice in detail['policies'].items():
                    require(all(identical(value, declared['policies'][policy].get(key)) for key, value in choice.items()),
                            'Selected policy metadata differs')
                choice = detail['policies'][POLICY]
                epoch = choice['selected_epoch']
                require(type(epoch) is int and 0 <= epoch <= training['epochs'], 'Invalid selected epoch')
                if not self.engineering:
                    require(epoch == spec['selected_epochs'][arm][str(seed)], 'Frozen P selected epoch changed')
                key = f'{name}/selected_{POLICY}.pt'
                path = {'source_checkpoint': key, 'source_checkpoint_sha256': choice['checkpoint_sha256'],
                        'selected_policy': POLICY, 'output_policy': POLICY, 'epoch': epoch,
                        'adapter_state_sha256': choice['adapter_state_sha256']}
                require(self.endpoints[key] == choice['checkpoint_sha256']
                        and identical(self.paths[f'{name}/{PATH}'], path), 'Selected P manifest identity mismatch')
                require(all(identical(declared['output_paths'][PATH].get(k), v) for k, v in path.items()),
                        'Output path identity differs from manifest')
                self.read(key)  # Hash only; never load last or rejected epoch states.
                require(self.hashes[key] == choice['checkpoint_sha256'], 'Checkpoint bytes disagree with frozen manifest')
                self.details[name] = detail

    def read(self, name, listed=True):
        path = relative_file(self.root, name)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if listed:
            require(self.summary['outputs'].get(name) == digest, f'Source artifact hash mismatch: {name}')
        require(name not in self.hashes or self.hashes[name] == digest, 'Source mutated during read')
        self.hashes[name] = digest
        return payload

    def array(self, name):
        if name not in self.records:
            with np.load(io.BytesIO(self.read(name)), allow_pickle=False) as values:
                self.records[name] = {key: values[key] for key in values.files}
        return self.records[name]

    def recheck(self):
        for name, digest in self.hashes.items():
            require(sha256(relative_file(self.root, name)) == digest, f'Source changed before publication: {name}')


def subset(record, size):
    return {key: value if key == 'regions' else value[:size] for key, value in record.items()}


def check_record(record):
    from experiments.chronological.vector_output_scope import _validate_record
    _validate_record(record)
    require(len(record['regions']) == 21, 'Require all original 21 regions')


def reconcile_metrics(record, declared):
    require(declared['samples'] == len(record['ids']), 'Saved metric sample count mismatch')
    for j, region in enumerate(record['regions']):
        point = geometry.regional.finite_number(geometry.regional.ratio(record['errors'][:, j].sum(), record['counts'][:, j].sum()))
        expected = declared['mae'][str(region)]
        require((point is None) == (expected is None), 'Saved metric availability mismatch')
        if point is not None:
            require(np.isclose(point, expected, rtol=1e-10, atol=1e-8), 'Saved NPZ and summary MAE differ')


def verify_arrays(source):
    """Verify all 63 original arrays before selecting engineering subsets."""
    s = source.summary
    for phase, cohorts in source.indices.items():
        for cohort, indices in cohorts.items():
            name = f'{phase}_{cohort}.npz'
            anchor = source.array('A/' + name)
            check_record(anchor)
            require(len(anchor['ids']) == len(indices), 'Saved cohort sample count changed')
            reconcile_metrics(anchor, s['baseline'][phase][cohort])
            for seed in source.seeds:
                for arm in source.protocol['arms']:
                    fit = f'{arm}_s{seed}'
                    value = source.array(f'{fit}/{PATH}/{name}')
                    check_record(value)
                    for key in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask', 'regions'):
                        require(np.array_equal(value[key], anchor[key]), f'Original A/P support mismatch: {key}')
                    for region in ('candidate_h7_h12', 'noncandidate_all'):
                        column = list(anchor['regions']).index(region)
                        require(np.array_equal(anchor['errors'][:, column], value['errors'][:, column]), 'Original P protection failed')
                    reconcile_metrics(value, s['runs'][str(seed)][fit]['output_paths'][PATH]['phase_metrics'][phase][cohort])
        if not source.engineering:
            weeks, weights = geometry.calendar(source.protocol['periods'][phase], source.protocol['bootstrap'])
            for seed in source.seeds:
                weekly = source.array(f'{phase}_weekly_s{seed}.npz')
                require(weekly['weeks'].tolist() == weeks, 'Saved calendar changed')
                for method, value in weights.items():
                    require(np.array_equal(weekly[method + '_weights'], value), 'Saved bootstrap draws changed')


def prepare_inputs(source, data_dir, primary_dir, secondary_dir, checkpoint):
    import torch
    from experiments.chronological import train_vector_output_scope as original
    base = original.base
    baseline_protocol = base.mechanisms.load_protocol(base.mechanisms.PROTOCOL)
    baseline, hashes = base.mechanisms.verify_inputs(data_dir, primary_dir, secondary_dir, checkpoint, baseline_protocol)
    fingerprint = lambda mapping: sorted((Path(name).name, digest) for name, digest in mapping.items())
    require(fingerprint(hashes) == fingerprint(source.summary['inputs']) and len(hashes) == 19,
            'All nineteen source input fingerprints must match')
    payload = ordinary(checkpoint).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == source.identity['checkpoint_sha256'], 'Original A identity changed')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    manifests = [base.read_csv(Path(root) / name) for root, name in (
        (data_dir, 'train_manifest.csv'), (primary_dir, 'train_control_manifest.csv'),
        (secondary_dir, 'train_second_control_manifest.csv'))]
    plan = base.make_plan(*manifests, source.inherited)
    require(identical(plan, source.plan), 'Recomputed eligibility differs from frozen source')
    positive = {int(row['sample_index']): (index, row) for index, row in enumerate(manifests[0])}
    first = {int(row['positive_sample_index']): row for row in manifests[1]}
    second = {int(row['positive_sample_index']): row for row in manifests[2]}
    metadata = {}
    for phase, cohorts in source.indices.items():
        for cohort, indices in cohorts.items():
            record = source.array(f'A/{phase}_{cohort}.npz')
            wanted = ([int(manifests[0][i]['sample_index']) for i in indices] if cohort == 'incident_full'
                      else [int(manifests[2][i]['positive_sample_index']) for i in indices])
            require(record['ids'].tolist() == wanted, 'Saved IDs disagree with manifest eligibility')
            rows = []
            for i, sample in enumerate(wanted):
                pos_index, pos = positive[sample]
                row = pos if cohort in ('incident_full', 'incident') else (first if cohort == 'primary_control' else second)[sample]
                source_index = pos_index if cohort in ('incident_full', 'incident') else int(row['control_index'])
                require(int(record['source_indices'][i]) == source_index, 'Cohort source index differs from manifest')
                rows.append({'incident_ids': pos['incident_id'], 'positive_t0': pos['t0'],
                    'cohort_t0': row['t0'] if cohort in ('incident_full', 'incident') else row['candidate_t0'],
                    'support_start': row['support_start'], 'support_end_exclusive': row['support_end_exclusive']})
            metadata[(phase, cohort)] = {key: np.asarray([row[key] for row in rows]) for key in geometry.META[2:]}
    return original, baseline, hashes, state, metadata


def assert_frozen(model, digest, original):
    import torch
    require(not model.training and not torch.is_grad_enabled(), 'Inference requires eval and no_grad')
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Parameter/gradient enabled')
    require(original.alignment.tensor_hash(model.state_dict()) == digest, 'Complete model/buffer state changed')


def load_selected(model, source, arm, seed, original):
    import torch
    name = f'{arm}_s{seed}'
    detail = source.details[name]
    choice = detail['policies'][POLICY]
    saved = torch.load(io.BytesIO(source.read(f'{name}/selected_{POLICY}.pt')), map_location='cpu', weights_only=True)
    native_hash = original.base.backbone_hash(model)
    adapter = original.vector.attach_adapter(model, arm, source.inherited['training']['node_hidden_width'])
    # The producer initialized after its seeded probe DataLoader consumed RNG.
    # This new adapter supplies only the required keys/shapes; its random initial
    # values are never a reference for the original saved initialization.
    initial_hash = detail['initial_adapter_sha256']
    identity = {'arm': arm, 'loss': 'candidate_early', 'seed': seed,
        'protocol_sha256': source.summary['protocol_sha256'],
        'run_identity_sha256': source.hashes['run_identity.json'], 'backbone_state_sha256': native_hash,
        'initial_adapter_sha256': initial_hash, 'output_policies': source.frozen['policies'],
        'fit_samples': len(source.indices['fit']['incident_full']), 'training': source.summary['effective_training']}
    require(set(saved) == {'identity', 'policy', 'epoch', 'adapter_state', 'adapter_state_sha256', 'selection_metrics'},
            'Selected checkpoint schema changed')
    require(identical(saved['identity'], identity) and saved['policy'] == POLICY
            and identical(saved['epoch'], choice['selected_epoch'])
            and identical(saved['selection_metrics'], choice['selection_metrics']), 'Selected checkpoint identity changed')
    require(detail['backbone_state_sha256'] == native_hash, 'Backbone state mismatch')
    state = saved['adapter_state']
    require(isinstance(state, dict) and set(state) == set(adapter.state_dict()), 'Invalid adapter state keys')
    for key, value in state.items():
        expected = adapter.state_dict()[key]
        require(isinstance(value, torch.Tensor) and value.shape == expected.shape and value.dtype == expected.dtype
                and torch.isfinite(value).all().item(), 'Invalid adapter tensor')
    digest = original.alignment.tensor_hash(state)
    require(digest == saved['adapter_state_sha256'] == choice['adapter_state_sha256'], 'Adapter tensor hash mismatch')
    adapter.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in adapter.parameters()) == detail['trainable_parameters'], 'Adapter parameter count changed')
    if saved['epoch'] == 0:
        require(digest == initial_hash, 'Epoch-zero adapter differs from A')
    model.requires_grad_(False).eval()
    original.base.assert_backbone(model, native_hash)
    return {'selected_epoch': saved['epoch'], 'backbone_state_sha256': native_hash,
            'adapter_state_sha256': digest, 'full_state_sha256': original.alignment.tensor_hash(model.state_dict())}


def replay(record, saved, protocol, labels):
    for key in protocol['replay']['exact_fields']:
        require(np.array_equal(record[key], saved[key]), f'Replay support differs: {key}')
    tolerance = protocol['replay']['per_window_error_sum_tolerance']
    require(np.isfinite(record['errors']).all() and np.allclose(record['errors'], saved['errors'], **tolerance),
            'Replay per-window regional errors exceed frozen tolerance')
    require(np.all(record['errors'][record['counts'] == 0] == 0), 'Nonzero empty-support errors')
    rows = []
    for region in REGIONS:
        j = list(saved['regions']).index(region)
        count = int(saved['counts'][:, j].sum())
        delta = record['errors'][:, j] - saved['errors'][:, j]
        diff = float(delta.sum() / count) if count else None
        require(diff is None or abs(diff) <= protocol['replay']['pooled_mae_and_gain_absolute_tolerance'],
                'Replay pooled MAE exceeds frozen tolerance')
        rows.append({**labels, 'region': region, 'valid_cells': count,
            'max_window_error_sum_difference': float(np.abs(delta).max()), 'pooled_mae_difference': diff})
    return rows


def pack_predictions(anchor, prediction, target, valid, reference, metadata):
    for array in (anchor, prediction, target):
        require(array.dtype == np.float32 and array.shape == valid.shape, 'Require aligned raw FP32 predictions/targets')
    require(valid.dtype == np.bool_ and valid.ndim == 4 and valid.shape[1] == 12 and valid.shape[3] == 1,
            'Require original twelve-step target validity')
    mask = reference['candidate_mask']
    require(mask.shape == (len(anchor), anchor.shape[2]), 'Candidate mask/forecast shape mismatch')
    offsets, nodes, horizons = [0], [], []
    cells = {key: [] for key in ('A', 'Y', 'd', 'valid')}
    for i, candidate in enumerate(mask):
        selected = np.flatnonzero(candidate)
        a = anchor[i, :6, :, 0][:, selected].reshape(-1)
        p = prediction[i, :6, :, 0][:, selected].reshape(-1)
        cells['A'].append(a); cells['Y'].append(target[i, :6, :, 0][:, selected].reshape(-1))
        cells['d'].append(p.astype(np.float64) - a.astype(np.float64))
        cells['valid'].append(valid[i, :6, :, 0][:, selected].reshape(-1))
        offsets.append(offsets[-1] + len(a))
        nodes.append(np.tile(selected, 6)); horizons.append(np.repeat(np.arange(1, 7, dtype=np.int64), len(selected)))
    return {**{key: np.concatenate(value) for key, value in cells.items()},
            'ids': reference['ids'].copy(), 'source_indices': reference['source_indices'].copy(),
            'candidate_mask': mask.copy(), 'window_offsets': np.asarray(offsets, dtype=np.int64),
            'candidate_node_indices': np.concatenate(nodes).astype(np.int64),
            'horizon_indices': np.concatenate(horizons), **metadata}


def infer_cohort(model, dataset, indices, original, device, progress, anchor=None):
    """Score A or projected P only. No U(eP) target error is calculated."""
    import torch
    base = original.base
    digest = original.alignment.tensor_hash(model.state_dict())
    pieces, predictions, targets, validity = {}, [], [], []
    position, batches = 0, 0
    with torch.no_grad():
        assert_frozen(model, digest, original)
        for raw in base.loader(dataset, indices, 16):
            batch = base.device_batch(raw, device)
            predicted = model(batch['x'], incident_data=batch['incident'])
            predicted = predicted * dataset.scaler['std'] + dataset.scaler['mean']
            if anchor is not None:
                a = torch.from_numpy(anchor[position:position + len(predicted)]).to(device)
                projected = original.scope.project_prediction(predicted, a, batch['candidate_mask'])
                active = base.masks_for(predicted, batch['candidate_mask'])[base.REGIONS.index('candidate_h1_h6')]
                require(torch.equal(projected[active], predicted[active]) and torch.equal(projected[~active], a[~active]),
                        'P prediction protection is not exact')
                predicted = projected
            values = base.statistics(predicted, batch)
            values.update(ids=raw['positive_sample_index'].numpy(), source_indices=raw['source_index'].numpy(),
                          candidate_mask=raw['candidate_mask'].numpy())
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
            predictions.append(predicted.cpu().numpy())
            targets.append(raw['y_flow'].numpy()); validity.append(raw['y_valid'].numpy())
            position += len(predicted); batches += 1
            if hasattr(model.icsf_module, 'clear_observations'):
                model.icsf_module.clear_observations()
            if batches % 10 == 0:
                progress('inference_progress', completed_windows=position, total_windows=len(indices))
        assert_frozen(model, digest, original)
    require(position == len(indices) and position > 0, 'Incomplete inference cohort')
    record = {**{key: np.concatenate(values) for key, values in pieces.items()}, 'regions': np.asarray(base.REGIONS)}
    return record, np.concatenate(predictions), np.concatenate(targets), np.concatenate(validity), batches


def new_output(requested, readonly_roots):
    output = ordinary(requested)
    partial = output.with_name(output.name + '.partial')
    for root in readonly_roots:
        root = ordinary(root)
        require(not output.is_relative_to(root) and not root.is_relative_to(output)
                and not partial.is_relative_to(root), 'Output must be outside read-only inputs')
    for path in (output, partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Preserve existing output/partial: {path}')
    partial.mkdir(parents=True)
    return output, partial


def progress_writer(partial):
    started = time.monotonic()
    def progress(stage, **fields):
        event = {'stage': stage, 'elapsed_seconds': time.monotonic() - started, **fields}
        write_json(partial / 'progress.json', event)
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
    return progress


def export_predictions(source, data_dir, primary_dir, secondary_dir, checkpoint, partial, protocol, device, check, progress):
    import torch
    initial_code = code_hashes(protocol)
    original, baseline, hashes, state, metadata = prepare_inputs(source, data_dir, primary_dir, secondary_dir, checkpoint)
    base = original.base
    torch.set_num_threads(3)
    device = torch.device(device)
    base.configure_determinism(device)
    datasets = base.make_datasets(data_dir, primary_dir, secondary_dir, baseline)
    def native():
        model = base.make_model(Path(data_dir), len(datasets['incident_full'].station_ids), device, 'fixed')
        model.load_state_dict(state, strict=True)
        require(sum(p.numel() for p in model.parameters()) == baseline['checkpoint']['parameters'], 'A parameter count changed')
        return model.requires_grad_(False).eval()
    seeds = source.seeds[:1] if check else source.seeds
    indices = {phase: {cohort: rows[:16] if check else rows for cohort, rows in cohorts.items()}
               for phase, cohorts in source.indices.items()}
    anchors, references, replay_rows, exported = {}, {}, [], []
    telemetry = {'native_forward_batches': 0, 'adapted_forward_batches': 0,
                 'optimizer_steps': 0, 'model_state_checks_passed': True}
    model = native()
    for phase, cohorts in indices.items():
        for cohort, rows in cohorts.items():
            progress('baseline_inference', phase=phase, cohort=cohort, windows=len(rows))
            saved = subset(source.array(f'A/{phase}_{cohort}.npz'), len(rows))
            record, prediction, _, _, batches = infer_cohort(model, datasets[cohort], rows, original, device, progress)
            replay_rows.extend(replay(record, saved, protocol, {'endpoint': 'A', 'phase': phase, 'cohort': cohort}))
            anchors[(phase, cohort)], references[(phase, cohort)] = prediction, record
            telemetry['native_forward_batches'] += batches
    del model
    selected = {}
    (partial / 'signed_cells').mkdir()
    for seed in seeds:
        for arm in protocol['arms']:
            name = f'{arm}_s{seed}'
            model = native()
            selected[name] = load_selected(model, source, arm, seed, original)
            progress('selected_endpoint_verified', endpoint=name, epoch=selected[name]['selected_epoch'])
            for phase, cohorts in indices.items():
                for cohort, rows in cohorts.items():
                    labels = {'endpoint': name, 'phase': phase, 'cohort': cohort}
                    progress('protected_inference', **labels, windows=len(rows))
                    saved = subset(source.array(f'{name}/{PATH}/{phase}_{cohort}.npz'), len(rows))
                    saved_a = subset(source.array(f'A/{phase}_{cohort}.npz'), len(rows))
                    ref, anchor = references[(phase, cohort)], anchors[(phase, cohort)]
                    record, predicted, y, valid, batches = infer_cohort(model, datasets[cohort], rows, original, device, progress, anchor)
                    telemetry['adapted_forward_batches'] += batches
                    replay_rows.extend(replay(record, saved, protocol, labels))
                    for key in ('candidate_mask', 'ids', 'source_indices', 'counts', 'prediction_counts'):
                        require(np.array_equal(ref[key], record[key]), 'Native/projected replay support changed')
                    for region in REGIONS:
                        j = list(record['regions']).index(region)
                        delta = (ref['errors'][:, j] - record['errors'][:, j]) - (saved_a['errors'][:, j] - saved['errors'][:, j])
                        count = int(ref['counts'][:, j].sum())
                        gap = float(delta.sum() / count) if count else None
                        require(gap is None or abs(gap) <= protocol['replay']['pooled_mae_and_gain_absolute_tolerance'],
                                'Replay G(1) exceeds frozen tolerance')
                        replay_rows.append({**labels, 'region': region, 'valid_cells': count,
                            'max_window_error_sum_difference': float(np.abs(delta).max()), 'pooled_gain_difference': gap})
                        if region in ('candidate_h7_h12', 'noncandidate_all'):
                            require(np.array_equal(record['errors'][:, j], ref['errors'][:, j]), 'Protected error differs from A')
                    meta = {key: value[:len(rows)] for key, value in metadata[(phase, cohort)].items()}
                    packed = pack_predictions(anchor, predicted, y, valid, ref, meta)
                    window = geometry.window_geometry(packed)
                    j = list(ref['regions']).index('candidate_h1_h6')
                    require(np.array_equal(window['valid_counts'], ref['counts'][:, j]), 'Signed export target support changed')
                    geometry.close(window['gain_sums'], ref['errors'][:, j] - record['errors'][:, j], 'Signed G differs from replay')
                    if not check and phase == 'audit' and cohort == 'incident_full':
                        require(window['gain_sums'].sum() < 0, 'Known negative audit G(1) reversed; stop interpretation')
                    filename = f'signed_cells/{name}__{phase}__{cohort}.npz'
                    np.savez_compressed(partial / filename, **packed)
                    exported.append({**labels, 'file': filename, 'windows': len(rows)})
            original.base.assert_backbone(model, selected[name]['backbone_state_sha256'])
            require(original.alignment.tensor_hash(model.state_dict()) == selected[name]['full_state_sha256'],
                    'Selected model changed across cohorts')
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    write_csv(partial / 'replay_checks.csv', replay_rows)
    (partial / 'protocol_snapshot.json').write_bytes(PROTOCOL.read_bytes())
    source.recheck()
    for name, digest in hashes.items():
        require(sha256(ordinary(name)) == digest, 'Raw input changed during inference')
    require(initial_code == code_hashes(protocol), 'Implementation changed during inference')
    inputs = {'raw_inputs': hashes, 'source_root': str(source.root), 'source_artifacts': source.hashes,
              'code_sha256': initial_code}
    write_json(partial / 'input_manifest.json', inputs)
    artifact_files = ['protocol_snapshot.json', 'input_manifest.json', 'replay_checks.csv'] + [item['file'] for item in exported]
    manifest = {'format_version': 'v12n_signed_export_v1', 'status': 'SIGNED_EXPORT_COMPLETE',
        'protocol_sha256': PROTOCOL_SHA256, 'engineering_check': check, 'engineering_source': source.engineering,
        'seeds': seeds, 'arms': protocol['arms'], 'entries': exported, 'selected_models': selected,
        'code_sha256': initial_code, 'telemetry': telemetry,
        'environment': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
            'device': str(device), 'host': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'memory_at_export': base.memory_snapshot(device)},
        'files': {name: sha256(partial / name) for name in artifact_files}}
    write_json(partial / 'signed_export_manifest.json', manifest)
    progress('signed_export_complete', endpoints=len(selected), files=len(exported), **telemetry)
    return manifest


def verified_export(root, protocol):
    root = ordinary(root)
    manifest = decode((root / 'signed_export_manifest.json').read_bytes())
    require(manifest.get('format_version') == 'v12n_signed_export_v1' and manifest.get('status') == 'SIGNED_EXPORT_COMPLETE'
            and manifest.get('protocol_sha256') == PROTOCOL_SHA256, 'Incomplete/wrong signed export')
    require(type(manifest.get('engineering_check')) is bool and type(manifest.get('engineering_source')) is bool,
            'Export lacks engineering boundary')
    require(not manifest['engineering_source'] or manifest['engineering_check'], 'Engineering export cannot become formal')
    require(manifest['code_sha256'] == code_hashes(protocol), 'Export implementation identity changed')
    seeds = protocol['seeds'][:1] if manifest['engineering_check'] else protocol['seeds']
    require(manifest['seeds'] == seeds and manifest['arms'] == protocol['arms'], 'Export endpoint set changed')
    endpoints = {f'{arm}_s{seed}' for seed in seeds for arm in protocol['arms']}
    require(set(manifest['selected_models']) == endpoints, 'Export selected model set changed')
    for seed in seeds:
        for arm in protocol['arms']:
            selected = manifest['selected_models'][f'{arm}_s{seed}']
            require(type(selected['selected_epoch']) is int, 'Invalid exported selected epoch')
            if not manifest['engineering_check']:
                require(selected['selected_epoch'] == protocol['source']['selected_epochs'][arm][str(seed)],
                        'Export selected epoch changed')
    expected = {(f'{arm}_s{seed}', phase, cohort) for seed in seeds for arm in protocol['arms']
                for phase, cohorts in protocol['expected_phase_samples'].items() for cohort in cohorts}
    observed = [(e['endpoint'], e['phase'], e['cohort']) for e in manifest['entries']]
    require(len(observed) == len(expected) and set(observed) == expected, 'Incomplete or duplicated signed export')
    wanted = {'protocol_snapshot.json', 'input_manifest.json', 'replay_checks.csv'}
    for item in manifest['entries']:
        expected_file = f"signed_cells/{item['endpoint']}__{item['phase']}__{item['cohort']}.npz"
        require(item['file'] == expected_file, 'Signed export path does not match identity')
        wanted.add(expected_file)
        n = protocol['expected_phase_samples'][item['phase']][item['cohort']]
        require(type(item['windows']) is int and (0 < item['windows'] <= min(16, n)
                if manifest['engineering_check'] else item['windows'] == n), 'Signed export sample budget changed')
    require(set(manifest['files']) == wanted, 'Export file allowlist changed')
    for name, digest in manifest['files'].items():
        require(sha256(relative_file(root, name)) == digest, f'Signed export hash mismatch: {name}')
    require(sha256(root / 'protocol_snapshot.json') == PROTOCOL_SHA256, 'Export protocol snapshot changed')
    return manifest


def statistics(export_root, partial, protocol, progress):
    manifest = verified_export(export_root, protocol)
    results, matched, metric_rows, categories, contrast_rows, sensitivity = {}, {}, [], [], [], []
    (partial / 'window_geometry').mkdir()
    calendars = {phase: geometry.calendar(bounds, protocol['bootstrap']) for phase, bounds in protocol['periods'].items()}
    for name in manifest['selected_models']:
        records, results[name], matched[name] = {}, {}, {}
        entries = [item for item in manifest['entries'] if item['endpoint'] == name]
        require(len(entries) == 9, 'Every endpoint must retain all nine phase/cohort entries')
        for item in entries:
            phase, cohort = item['phase'], item['cohort']
            with np.load(relative_file(export_root, item['file']), allow_pickle=False) as arrays:
                packed = {key: arrays[key] for key in arrays.files}
            require(len(packed['ids']) == item['windows'], 'Export declared sample count changed')
            record = geometry.window_geometry(packed)
            records[(phase, cohort)] = record
            np.savez_compressed(partial / 'window_geometry' / Path(item['file']).name, **record)
            weeks, weights = calendars[phase]
            result, rows = geometry.analyze_windows(record, weeks, weights, protocol)
            results[name].setdefault(phase, {})[cohort] = result
            labels = {'endpoint': name, 'phase': phase, 'cohort': cohort}
            categories.extend({**labels, **row} for row in rows)
            for estimand, detail in result['estimands'].items():
                metric_rows.append({**labels, 'estimand': estimand, **detail['point'], 'description': detail['description'],
                    **{k: result[k] for k in ('forecast_windows', 'unique_incidents', 'valid_cells', 'evaluable_windows', 'zero_support_windows')},
                    **{f'{method}_{key}_{bound}': interval[bound] for method, values in detail['intervals'].items()
                       for key, interval in values.items() for bound in ('status', 'valid_draws', 'ci_low', 'ci_high')}})
        for phase in ('selection', 'audit'):
            weeks, weights = calendars[phase]
            value, rows = geometry.analyze_matched({cohort: records[(phase, cohort)] for cohort in geometry.COHORTS},
                                                  weeks, weights, protocol)
            matched[name][phase] = value
            sensitivity.extend({'endpoint': name, 'phase': phase, **row} for row in rows)
            for contrast, points in value['points'].items():
                for estimand, point in points.items():
                    contrast_rows.append({'endpoint': name, 'phase': phase, 'contrast': contrast, 'estimand': estimand,
                        'gain_difference': point, 'triplets': value['triplets'], 'complete_triplets': value['complete_triplets'],
                        'excluded_triplets': value['excluded_triplets'],
                        **{f'{cohort}_{key}': val for cohort, coverage in value['coverage'].items() for key, val in coverage.items()},
                        **{f'{method}_{bound}': interval[estimand][bound] for method, interval in value['intervals'][contrast].items()
                           for bound in ('status', 'valid_draws', 'ci_low', 'ci_high')}})
        if not manifest['engineering_check']:
            primary = results[name]['audit']['incident_full']['estimands']
            require(primary['pooled_valid_cells']['point']['gain'] is not None
                    and primary['pooled_valid_cells']['point']['gain'] < 0, 'Known negative G(1) reversed in signed export')
        progress('endpoint_statistics_complete', endpoint=name)
    for filename, rows in (('geometry_metrics.csv', metric_rows), ('category_contributions.csv', categories),
                           ('matched_contrasts.csv', contrast_rows), ('matched_week_sensitivity.csv', sensitivity)):
        write_csv(partial / filename, rows)
    verified_export(export_root, protocol)
    engineering = manifest['engineering_check']
    summary = {'status': 'ENGINEERING_CHECK_PASS' if engineering else 'VECTOR_CORRECTION_GEOMETRY_AUDIT_COMPLETE',
        'protocol_id': protocol['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256, 'engineering_check': engineering,
        'scientific_status': 'NOT_EVALUATED_ENGINEERING_ONLY' if engineering else 'FIXED_SAMPLE_DIAGNOSTIC_REQUIRES_REVIEW',
        'recommendation': 'NO_NEW_TRAINING_SELECTION_OR_TEST_AUTHORIZATION',
        'model_training_performed': False, 'gradient_computation_performed': False, 'optimizer_created': False,
        'new_model_selection_performed': False, 'lambda_search_performed': False,
        'validation_arrays_read': False, 'test_split_read': False, 'independent_confirmation': False,
        'legacy_A_scaler_information_dependencies_retained': True,
        'results': results, 'matched': matched, 'selected_models': manifest['selected_models'],
        'primary_category_contributions': [row for row in categories
            if row['phase'] == 'audit' and row['cohort'] == 'incident_full'],
        'signed_export_manifest_sha256': sha256(export_root / 'signed_export_manifest.json'),
        'export_environment': manifest['environment'], 'inference_telemetry': manifest['telemetry'],
        'code_sha256': code_hashes(protocol),
        'limitations': ['Post-hoc fixed predictions and repeatedly reused development; nine audit weeks.',
            'S is sensitive to near-zero residuals. Numeric reporting band is not a rigorous error certificate.',
            'Pointwise intervals do not give joint mechanism confirmation or independent seed replications.',
            'Routine pairing by incident week does not capture all cross-control-week traffic dependence.',
            'No lambda was chosen; final-output scaling theorem does not cover retraining or internal adapter scaling.']}
    (partial / 'report.md').write_text(report_text(summary), encoding='utf-8')
    return summary


def report_text(summary):
    lines = [f"# v12n: {summary['status']}", '', f"Scientific status: {summary['scientific_status']}",
             'NO NEW TRAINING / NO LAMBDA SEARCH / NO TEST AUTHORIZATION', '']
    if summary['engineering_check']:
        lines.append('Engineering subset only. These values do not establish any scientific conclusion.')
    else:
        lines.extend(['Primary: audit / incident_full / candidate_h1_h6; gain=A MAE-P MAE.', '',
            '| Endpoint | Weighting | G(1) [four-week 95% CI] | S [four-week 95% CI] | O [four-week 95% CI] | Description |',
            '| --- | --- | --- | --- | --- | --- |'])
        for name, phases in summary['results'].items():
            for estimand, detail in phases['audit']['incident_full']['estimands'].items():
                columns = []
                for key in geometry.QUANTITIES:
                    ci = detail['intervals']['four_week_block'][key]
                    columns.append(f"{detail['point'][key]} [{ci['ci_low']}, {ci['ci_high']}] ({ci['status']})")
                lines.append(f"| {name} | {estimand} | " + ' | '.join(columns) + f" | {detail['description']} |")
        lines.extend(['', 'Fixed-endpoint phase context (pooled full incidents; no cross-phase paired CI):', '',
            '| Endpoint | Phase | G(1) | S | O |', '| --- | --- | ---: | ---: | ---: |'])
        for name, phases in summary['results'].items():
            for phase in ('fit', 'selection'):
                p = phases[phase]['incident_full']['estimands']['pooled_valid_cells']['point']
                lines.append(f"| {name} | {phase} | {p['gain']} | {p['slope']} | {p['crossing_penalty']} |")
        lines.extend(['', 'Matched incident minus controls, audit (gain differences, not causal effects):', '',
            '| Endpoint | Contrast | Weighting | D | Four-week 95% CI | Complete / all triplets |',
            '| --- | --- | --- | ---: | --- | --- |'])
        for name, phases in summary['matched'].items():
            value = phases['audit']
            for contrast, points in value['points'].items():
                for estimand, point in points.items():
                    ci = value['intervals'][contrast]['four_week_block'][estimand]
                    lines.append(f"| {name} | {contrast} | {estimand} | {point} | [{ci['ci_low']}, {ci['ci_high']}] "
                                 f"({ci['status']}) | {value['complete_triplets']} / {value['triplets']} |")
        lines.extend(['', 'Primary pooled category contributions (full-cohort denominator):', '',
            '| Endpoint | Category | Cells | G contribution | S contribution | O contribution |',
            '| --- | --- | ---: | ---: | ---: | ---: |'])
        for row in summary['primary_category_contributions']:
            if row['estimand'] == 'pooled_valid_cells':
                lines.append(f"| {row['endpoint']} | {row['category']} | {row['cells']} | {row['gain_contribution']} | "
                             f"{row['slope_contribution']} | {row['crossing_penalty_contribution']} |")
        lines.extend(['', 'All phase/cohort G/S/O, ordinary-week and four-week intervals: geometry_metrics.csv; matched differences: matched_contrasts.csv.',
                      'Category contributions: category_contributions.csv; any-member support-week deletions: matched_week_sensitivity.csv.'])
    lines.extend(['', *summary['limitations'], ''])
    return '\n'.join(lines)


def run(source_dir, data_dir, primary_dir, secondary_dir, checkpoint, output, device='cuda:0',
        check=False, allow_engineering_source=False):
    require(not allow_engineering_source or check, 'Engineering source is allowed only with --check')
    protocol = load_protocol()
    output, partial = new_output(output, (source_dir, data_dir, primary_dir, secondary_dir, Path(checkpoint).parent))
    progress = progress_writer(partial)
    try:
        progress('verifying_source')
        source = Source(source_dir, protocol, allow_engineering=allow_engineering_source)
        verify_arrays(source)
        progress('source_arrays_verified', read_artifacts=len(source.hashes))
        export_predictions(source, data_dir, primary_dir, secondary_dir, checkpoint, partial, protocol, device, check, progress)
        summary = statistics(partial, partial, protocol, progress)
        source.recheck()
        summary['new_model_inference_this_invocation'] = True
        summary['outputs'] = {str(p.relative_to(partial)): sha256(p) for p in partial.rglob('*')
                              if p.is_file() and p.name != 'progress.json'}
        write_json(partial / 'summary.json', summary)
        partial.rename(output)
        print(report_text(summary), flush=True)
        return summary
    except Exception as error:
        write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise


def analyze_export(source_dir, output):
    protocol = load_protocol()
    source = ordinary(source_dir)
    manifest = verified_export(source, protocol)
    output, partial = new_output(output, (source,))
    progress = progress_writer(partial)
    try:
        for name in (*manifest['files'], 'signed_export_manifest.json'):
            path = relative_file(partial, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(relative_file(source, name), path)
        summary = statistics(partial, partial, protocol, progress)
        verified_export(source, protocol)
        summary['new_model_inference_this_invocation'] = False
        summary['statistics_source'] = str(source)
        summary['outputs'] = {str(p.relative_to(partial)): sha256(p) for p in partial.rglob('*')
                              if p.is_file() and p.name != 'progress.json'}
        write_json(partial / 'summary.json', summary)
        partial.rename(output)
        print(report_text(summary), flush=True)
        return summary
    except Exception as error:
        write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'preflight'):
        command = commands.add_parser(name)
        command.add_argument('--source', required=True)
        command.add_argument('--data-dir', required=True)
        command.add_argument('--primary-control-dir', required=True)
        command.add_argument('--secondary-control-dir', required=True)
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--check', action='store_true')
        command.add_argument('--allow-engineering-source', action='store_true')
        if name == 'run':
            command.add_argument('--output', required=True)
            command.add_argument('--device', default='cuda:0')
    command = commands.add_parser('analyze')
    command.add_argument('--export-dir', required=True)
    command.add_argument('--output', required=True)
    command = commands.add_parser('report')
    command.add_argument('summary')
    args = parser.parse_args()
    if args.command == 'report':
        summary = decode(ordinary(args.summary).read_bytes())
        require(summary['protocol_sha256'] == PROTOCOL_SHA256 and summary['status'] in
                ('ENGINEERING_CHECK_PASS', 'VECTOR_CORRECTION_GEOMETRY_AUDIT_COMPLETE'), 'Not a completed v12n result')
        print(report_text(summary))
    elif args.command == 'analyze':
        analyze_export(args.export_dir, args.output)
    elif args.command == 'preflight':
        require(not args.allow_engineering_source or args.check, 'Engineering source requires --check')
        source = Source(args.source, load_protocol(), allow_engineering=args.allow_engineering_source)
        verify_arrays(source)
        prepare_inputs(source, args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.checkpoint)
        source.recheck()
        print('PREFLIGHT_PASS: source identities, arrays and original train inputs verified; no model inference.')
    else:
        run(args.source, args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.checkpoint,
            args.output, args.device, args.check, args.allow_engineering_source)


if __name__ == '__main__':
    main()
