"""Train a small gain predictor on a certified training-internal holdout only.

The label is |y-base|-|y-candidate|, not candidate error or improvement
probability. The fixed ridge estimator predicts conditional *mean* MAE gain.
Positive predicted gain adopts the candidate; every other point keeps the base.
This does not guarantee individual non-harm, calibrated confidence, independent
events, or whole-pipeline out-of-fold predictions. Validation labels never fit
the head, its threshold, its features, or its regularization.
"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.chronological.evaluate_unified_adapter_v03 import (
    ARCHIVE_SCHEMA, AXIS_ORDER, analyze_archive, array_hash, paired_metrics, point_metrics,
    require, validate_archive,
)


SCHEMA = 'igstgnn_unified_adapter_gain_head_v03'
ALLOWED_SOURCES = {'history', 'report', 'model_prediction', 'model_hidden', 'clock', 'sensor', 'static', 'node', 'horizon'}


def _mapping(archive):
    if isinstance(archive, (str, Path)):
        with np.load(archive, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    return archive


def validate_archive_digest(source, metadata):
    """Bind file-backed inputs to producer-published bytes; mappings are fixtures/API inputs."""
    require(metadata.get('producer_status') == 'PASS', 'Producer must finish replay and publish PASS before auxiliary fitting')
    if isinstance(source, (str, Path)):
        expected = metadata.get('archive_sha256')
        require(isinstance(expected, str) and expected == sha256(Path(source).read_bytes()).hexdigest(),
                'Archive file does not match the producer-published SHA256')


def validate_prediction_roles(metadata, base_case, candidate_case):
    require(base_case == 'H_ONLY' and candidate_case == 'BOTH',
            'This protocol binds the fallback H_ONLY and candidate BOTH; other case bindings need a new protocol')
    roles = metadata.get('prediction_case_roles', {})
    require(isinstance(roles, dict), 'Prediction case roles must be a mapping')
    for case, field in ((base_case, 'base_checkpoint_sha256'), (candidate_case, 'candidate_checkpoint_sha256')):
        role = roles.get(case)
        require(isinstance(role, dict) and isinstance(role.get('checkpoint_sha256'), str)
                and bool(role['checkpoint_sha256']) and role['checkpoint_sha256'] == metadata.get(field),
                'Concrete '+case+' role checkpoint hash must match '+field)
    return {case: dict(roles[case]) for case in (base_case, candidate_case)}


def timed_cpu_call(function, *args, **kwargs):
    """Wall/process CPU timing includes helper input validation; no model forward."""
    wall, cpu = time.perf_counter(), time.process_time()
    result = function(*args, **kwargs)
    return result, dict(wall_seconds=time.perf_counter()-wall, cpu_process_seconds=time.process_time()-cpu,
                        scope='auxiliary_CPU_helper_includes_archive_IO_validation_excludes_all_source_model_forwards')


def validate_holdout_identity(calibration, validation, calibration_metadata, validation_metadata, base_case, candidate_case):
    """Check asserted sample-level provenance, without inventing event identity."""
    c, v = calibration_metadata, validation_metadata
    require(c.get('split') == 'calibration' and v.get('split') == 'validation',
            'Gain fitting requires a training-internal calibration split and separate validation')
    for metadata in (c, v):
        require(metadata.get('producer_status') == 'PASS', 'Producer replay must PASS before auxiliary fitting')
        for field in ('sample_identity_verified', 'eligible_for_gain_fit', 'candidate_frozen_before_calibration'):
            require(metadata.get(field) is True, 'Missing required provenance: '+field)
        for field in ('calibration_used_for_candidate_fit', 'calibration_used_for_candidate_selection', 'test_evaluated'):
            require(metadata.get(field) is False, 'Unsafe or missing provenance: '+field)
        require(metadata.get('candidate_checkpoint_sha256') and metadata.get('base_checkpoint_sha256'),
                'Both frozen checkpoint identities are required')
        validate_prediction_roles(metadata, base_case, candidate_case)
    for key in ('candidate_checkpoint_sha256', 'base_checkpoint_sha256', 'prediction_case_roles'):
        require(c[key] == v[key], 'Calibration and validation must use the same frozen prediction identity: '+key)
    train_c, train_v = c.get('candidate_training_sample_ids'), v.get('candidate_training_sample_ids')
    require(isinstance(train_c, list) and len(train_c) and isinstance(train_v, list),
            'Explicit candidate training identities are required, not only a split size')
    require(len(set(map(str, train_c))) == len(train_c) and set(map(str, train_c)) == set(map(str, train_v)),
            'Candidate training identity differs across archives or contains duplicates')
    train_ids = set(map(str, train_c))
    cal_ids, val_ids = set(map(str, calibration['sample_ids'])), set(map(str, validation['sample_ids']))
    require(not (train_ids & cal_ids or train_ids & val_ids or cal_ids & val_ids),
            'Candidate fit, calibration, and validation sample identities must be disjoint')
    base_training = c.get('base_training_sample_ids')
    require(isinstance(base_training, list) and base_training == v.get('base_training_sample_ids'),
            'Base training sample provenance is required as well as candidate training provenance')
    require(not (set(map(str, base_training)) & (cal_ids | val_ids)),
            'The learned fallback must not be fitted on gain calibration or validation samples')
    require(np.array_equal(calibration['node_ids'], validation['node_ids']), 'Node identities must agree across splits')
    group_verified = c.get('group_identity_verified') is True and v.get('group_identity_verified') is True
    if group_verified:
        fields = ('candidate_training_group_ids', 'base_training_group_ids', 'calibration_group_ids', 'validation_group_ids')
        require(all(isinstance(c.get(key), list) and c[key] == v.get(key) and len(c[key]) for key in fields),
                'Declared group split provenance must contain candidate/base training, calibration, and validation lists')
        groups = [set(map(str, c[key])) for key in fields]
        require(not (groups[0] & (groups[2] | groups[3]) or groups[1] & (groups[2] | groups[3]) or groups[2] & groups[3]),
                'Declared candidate/base training groups must be disjoint from calibration and validation groups')
    return dict(identity_scope='producer_declared_groups' if group_verified else 'sample_ids_only_overlap_not_certified',
                declared_groups_disjoint=group_verified,
                whole_pipeline_oof=False, independent_events_certified=False,
                note='producer provenance is checked for consistency, not independently reconstructed')


def encode_feature(value, record, name):
    """One-hot declared categories, never treat numeric IDs as ordinal semantics."""
    cardinalities = record.get('categorical_cardinalities')
    if cardinalities is None:
        require(record.get('information_channel') != 'C' or record.get('encoding') in ('learned_embedding', 'one_hot'),
                'Semantic category IDs require declared one-hot cardinalities; no ordinal regression: '+name)
        return value.astype(np.float64), [name+':'+str(index) for index in range(value.shape[-1])]
    require(isinstance(cardinalities, list) and len(cardinalities) == value.shape[-1]
            and all(isinstance(size, int) and size > 1 for size in cardinalities),
            'Categorical cardinalities must identify every channel: '+name)
    encoded, names = [], []
    for index, size in enumerate(cardinalities):
        column = value[:, index]
        require(np.isfinite(column).all() and (column == np.floor(column)).all()
                and (column >= 0).all() and (column < size).all(),
                'Category IDs must be legal integers including an explicit UNK category: '+name)
        encoded.append((column[:, None] == np.arange(size)[None, :]).astype(np.float64))
        names.extend(name+':'+str(index)+'='+str(category) for category in range(size))
    return np.concatenate(encoded, -1), names


def feature_matrix(raw, checked, metadata, base_case, candidate_case, valid_only=True, flat_indices=None):
    """Extract only explicitly allowed prediction-time features, in stable order."""
    base, candidate = checked['predictions'][base_case], checked['predictions'][candidate_case]
    indices = (np.flatnonzero(checked['valid']) if valid_only else np.arange(base.size)) if flat_indices is None else flat_indices
    b, c = base.reshape(-1)[indices], candidate.reshape(-1)[indices]
    features = [b[:, None], c[:, None], (c-b)[:, None]]
    names = ['base_prediction', 'candidate_prediction', 'proposed_change']
    manifest = metadata.get('feature_manifest', {})
    require(isinstance(manifest, dict), 'feature_manifest must identify archived feature provenance')
    for key in sorted(key for key in raw if key.startswith('feature__')):
        name = key[9:]
        require(name in manifest and manifest[name].get('source') in ALLOWED_SOURCES,
                'Feature source is not certified as prediction-time available: '+name)
        require(manifest[name].get('available_at_cutoff') is True,
                'Feature must be available at prediction cutoff: '+name)
        array = np.asarray(raw[key])
        require(array.shape == checked['y'].shape or (array.ndim == 4 and array.shape[:3] == checked['y'].shape),
                'Features must follow exact [sample,12,node,optional_feature] axes: '+name)
        require(array.dtype.kind in 'fiu', 'Decision features must be numeric: '+name)
        if array.ndim == 3:
            array = array[..., None]
        require(array.shape[-1] > 0, 'Empty decision feature')
        selected = array.reshape(-1, array.shape[-1])[indices]
        require(np.isfinite(selected).all(), 'Decision features must be finite: '+name)
        encoded, labels = encode_feature(selected, manifest[name], name)
        features.append(encoded)
        names.extend(labels)
    require(len(names) <= 128, 'First-stage gain head supports at most 128 explicitly documented features')
    x = np.concatenate(features, axis=-1).astype(np.float64)
    require(np.isfinite(x).all(), 'All decision inputs must be finite, including non-evaluated target positions')
    return x, names


def fit_gain_head(calibration_archive, validation_archive, calibration_metadata, validation_metadata,
                  base_case='H_ONLY', candidate_case='BOTH', regularization=1.):
    """Fit fixed ridge on calibration labels; validation y is never a fit input."""
    require(np.isfinite(regularization) and regularization > 0, 'Ridge regularization must be fixed and positive')
    validate_archive_digest(calibration_archive, calibration_metadata)
    validate_archive_digest(validation_archive, validation_metadata)
    raw_c, raw_v = _mapping(calibration_archive), _mapping(validation_archive)
    cal, val = validate_archive(raw_c), validate_archive(raw_v)
    require(all(name in cal['predictions'] and name in val['predictions'] for name in (base_case, candidate_case)),
            'Both splits require the same fallback and candidate cases')
    identity = validate_holdout_identity(cal, val, calibration_metadata, validation_metadata, base_case, candidate_case)
    require(calibration_metadata.get('feature_manifest', {}) == validation_metadata.get('feature_manifest', {}),
            'Feature definitions must be frozen before validation')
    eligible = np.flatnonzero(cal['valid'])
    fit_indices = eligible if len(eligible) <= 200000 else np.sort(np.random.default_rng(2025).choice(eligible, 200000, replace=False))
    x, names = feature_matrix(raw_c, cal, calibration_metadata, base_case, candidate_case, flat_indices=fit_indices)
    # Validation features establish shape/identity only, never normalization or fitting.
    _, val_names = feature_matrix(raw_v, val, validation_metadata, base_case, candidate_case, flat_indices=np.array([0]))
    require(names == val_names, 'Calibration and validation feature axes disagree')
    require(len(x) >= max(2, len(names)+1), 'Too few eligible calibration points for the fixed gain head')
    target = cal['y'].reshape(-1)[fit_indices].astype(np.float64)
    base = cal['predictions'][base_case].reshape(-1)[fit_indices].astype(np.float64)
    candidate = cal['predictions'][candidate_case].reshape(-1)[fit_indices].astype(np.float64)
    gain = np.abs(target-base)-np.abs(target-candidate)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.
    z = (x-mean)/scale
    gain_mean = float(gain.mean())
    # Divide by sample count so the fixed penalty does not depend on point count.
    covariance = z.T @ z/len(z)
    cross = z.T @ (gain-gain_mean)/len(z)
    weights = np.linalg.solve(covariance+regularization*np.eye(z.shape[1]), cross)
    require(np.isfinite(weights).all(), 'Gain head fitting produced nonfinite coefficients')
    return dict(schema=SCHEMA, base_case=base_case, candidate_case=candidate_case,
                feature_names=names, feature_manifest=calibration_metadata.get('feature_manifest', {}),
                mean=mean.tolist(), scale=scale.tolist(), weights=weights.tolist(), intercept=gain_mean,
                ridge_regularization=float(regularization), threshold=0.,
                fit_split='calibration', fit_points=len(x), eligible_points=len(eligible),
                point_subsample_seed=2025, point_subsample_cap=200000, fit_sample_ids=cal['sample_ids'].tolist(),
                fit_sample_ids_sha256=array_hash(cal['sample_ids']), identity=identity,
                candidate_checkpoint_sha256=calibration_metadata['candidate_checkpoint_sha256'],
                base_checkpoint_sha256=calibration_metadata['base_checkpoint_sha256'],
                prediction_case_roles=validate_prediction_roles(calibration_metadata, base_case, candidate_case),
                labels='absolute_error_base_minus_absolute_error_candidate',
                validation_labels_used_for_fitting=False, test_evaluated=False)


def predict_gain_head(model, archive, metadata):
    require(model.get('schema') == SCHEMA, 'Unsupported gain head schema')
    validate_archive_digest(archive, metadata)
    require(validate_prediction_roles(metadata, model['base_case'], model['candidate_case']) == model['prediction_case_roles'],
            'Concrete prediction case roles changed after gain fitting')
    require(metadata.get('candidate_checkpoint_sha256') == model['candidate_checkpoint_sha256']
            and metadata.get('base_checkpoint_sha256') == model['base_checkpoint_sha256'],
            'Never reuse a gain head after candidate/base checkpoint changes')
    require(metadata.get('feature_manifest', {}) == model['feature_manifest'], 'Feature provenance changed after fitting')
    raw = _mapping(archive)
    checked = validate_archive(raw)
    prediction = np.empty(checked['y'].size, np.float64)
    for start in range(0, len(prediction), 65536):
        indices = np.arange(start, min(start+65536, len(prediction)))
        x, names = feature_matrix(raw, checked, metadata, model['base_case'], model['candidate_case'], flat_indices=indices)
        require(names == model['feature_names'], 'Feature axis changed after fitting')
        prediction[indices] = ((x-np.asarray(model['mean']))/np.asarray(model['scale'])) @ np.asarray(model['weights'])+model['intercept']
    require(np.isfinite(prediction).all(), 'Predicted gains must be finite')
    return prediction.reshape(checked['y'].shape)


def evaluate_policies(base, candidate, y, valid, predicted_gain, seed=2025, global_adopt=None):
    """Descriptive all-point controls; oracle is explicitly undeployable.

    The matched-random control exactly matches learned acceptance among saved
    valid points. It uses the evaluation mask for *diagnostic matching*, and is
    not a deployment rule. No threshold is selected using validation y.
    """
    require(base.shape == candidate.shape == y.shape == valid.shape == predicted_gain.shape,
            'Policy arrays must have exactly the shared prediction axes')
    require(valid.dtype == np.bool_ and np.isfinite(predicted_gain).all(), 'Invalid policy mask or predicted gains')
    learned = predicted_gain > 0
    random = np.zeros(y.size, bool)
    eligible = np.flatnonzero(valid.ravel())
    accepted = int(learned[valid].sum())
    random[np.random.default_rng(seed).permutation(eligible)[:accepted]] = True
    random = random.reshape(y.shape)
    # Oracle labels are accessed only here, after the learned policy is fixed.
    oracle = np.abs(y-base) > np.abs(y-candidate)
    masks = dict(ALWAYS_BASE=np.zeros(y.shape, bool), ALWAYS_CANDIDATE=np.ones(y.shape, bool),
                 LEARNED=learned, RANDOM_MATCHED=random, ORACLE_UNDEPLOYABLE=oracle)
    if global_adopt is not None:
        masks['GLOBAL_CALIBRATION'] = np.full(y.shape, bool(global_adopt), bool)
    predictions = {name: np.where(mask, candidate, base) for name, mask in masks.items()}
    predictions['HALF_CORRECTION'] = base+.5*(candidate-base)
    results = {}
    for name, prediction in predictions.items():
        results[name] = dict(point_metrics(prediction, y, valid),
                             gain_vs_base=paired_metrics(base, prediction, y, valid),
                             acceptance_fraction=float(masks[name][valid].mean()) if name in masks else None,
                             deployable=name not in ('ORACLE_UNDEPLOYABLE', 'RANDOM_MATCHED'),
                             learned_policy_degenerate=bool((learned[valid].all() or not learned[valid].any())) if name == 'LEARNED' else None)
    return results, predictions


def run_decision_analysis(calibration_archive, validation_archive, calibration_metadata, validation_metadata,
                          output_dir, base_case='H_ONLY', candidate_case='BOTH'):
    model, fit_timing = timed_cpu_call(fit_gain_head, calibration_archive, validation_archive, calibration_metadata, validation_metadata,
                                      base_case, candidate_case)
    gain, predict_timing = timed_cpu_call(predict_gain_head, model, validation_archive, validation_metadata)
    raw = _mapping(validation_archive)
    checked = validate_archive(raw)
    policies, predictions = evaluate_policies(checked['predictions'][base_case], checked['predictions'][candidate_case],
                                             checked['y'], checked['valid'], gain, global_adopt=model['intercept'] > 0)
    output = Path(output_dir)
    require(not output.exists() or not any(output.iterdir()), 'Decision output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema=SCHEMA, status='GAIN_HEAD_DESCRIPTIVE_VALIDATION_COMPLETE', model=model, policies=policies,
                  runtime=dict(fit=dict(fit_timing, processed_points=model['fit_points'], iterations=1,
                                        algorithm='closed_form_ridge_normal_equations'),
                               predict=dict(predict_timing, processed_points=int(gain.size), iterations=math.ceil(gain.size/65536)),
                               deployment_required_source_model_forwards=2,
                               deployment_source_models=['H_ONLY', 'BOTH'],
                               complete_deployment_latency_measured=False,
                               note='LEARNED requires both separately trained model forecasts plus feature and gain-head work; auxiliary CPU time is not full model latency'),
                  boundaries=['sample_holdout_does_not_certify_independent_overlapping_events',
                              'no_pointwise_nonharm_guarantee', 'no_validation_fit_or_threshold_search',
                              'random_matching_uses_only_original_valid_mask_not_a_deployment_rule',
                              'oracle_uses_truth_is_undeployable_not_a_candidate',
                              'degenerate_all_open_or_all_closed_is_not_evidence_of_selective_value'],
                  test_evaluated=False)
    archived = {key: value for key, value in raw.items() if not key.startswith('pred__') and not key.startswith('q__')}
    archived.update({'schema': np.array(ARCHIVE_SCHEMA), 'axis_order': np.array(AXIS_ORDER),
                     'pred__ORIGINAL': checked['predictions']['ORIGINAL'],
                     'pred__H_ONLY': checked['predictions'][base_case], 'pred__BOTH': checked['predictions'][candidate_case],
                     'predicted_gain': gain})
    archived.update({'pred__'+name: prediction for name, prediction in predictions.items()})
    # A gated quantile result needs the same decision applied to base/candidate
    # quantiles; do not silently pair an adopted point with ungated intervals.
    if base_case in checked['quantiles'] and candidate_case in checked['quantiles']:
        from experiments.chronological.evaluate_unified_adapter_v03 import quantile_metrics
        q = np.where((gain > 0)[..., None], checked['quantiles'][candidate_case], checked['quantiles'][base_case])
        report['learned_interval'] = quantile_metrics(q, checked['y'], checked['valid'])
        archived['q__LEARNED'] = q
        archived['quantile_levels'] = np.array([.1, .5, .9])
    np.savez_compressed(output/'decision_predictions.npz', **archived)
    analyze_archive(output/'decision_predictions.npz', output/'analysis', metadata=validation_metadata)
    report['paired_analysis_directory'] = 'analysis'
    (output/'decision.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-archive', required=True, type=Path)
    parser.add_argument('--validation-archive', required=True, type=Path)
    parser.add_argument('--calibration-metadata', required=True, type=Path)
    parser.add_argument('--validation-metadata', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--base-case', default='H_ONLY')
    parser.add_argument('--candidate-case', default='BOTH')
    args = parser.parse_args(argv)
    report = run_decision_analysis(args.calibration_archive, args.validation_archive,
                                   json.loads(args.calibration_metadata.read_text(encoding='utf-8')),
                                   json.loads(args.validation_metadata.read_text(encoding='utf-8')),
                                   args.output_dir, args.base_case, args.candidate_case)
    print(json.dumps(dict(schema=SCHEMA, status=report['status'], validation_fit=False, test_evaluated=False), indent=2))


if __name__ == '__main__':
    main()
