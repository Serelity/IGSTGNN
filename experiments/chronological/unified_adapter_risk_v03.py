"""Conditional .1/.9 interval heads around the same frozen H-only point forecast.

The three H/H_R/H_R_C heads address report value for risk without letting report
inputs move the center. Fixed small linear heads learn positive softplus half
widths with pinball loss on a training-internal holdout. They are not an exact
predictive distribution, conformal calibration, or a guarantee of 80% coverage.
"""
from argparse import ArgumentParser
import json
import math
from pathlib import Path

import numpy as np

from experiments.chronological.evaluate_unified_adapter_v03 import analyze_archive, quantile_metrics, require, validate_archive
from experiments.chronological.unified_adapter_decision_v03 import (
    ALLOWED_SOURCES, _mapping, encode_feature, timed_cpu_call, validate_archive_digest,
    validate_holdout_identity, validate_prediction_roles,
)


SCHEMA = 'igstgnn_unified_adapter_risk_head_v03'
MODES = {'H': {'H'}, 'H_R': {'H', 'R'}, 'H_R_C': {'H', 'R', 'C'}}


def _features(raw, checked, metadata, center_case, mode, valid_only, flat_indices=None):
    require(mode in MODES, 'Risk mode must be H, H_R, or H_R_C')
    channels = MODES[mode]
    center = checked['predictions'][center_case]
    indices = (np.flatnonzero(checked['valid']) if valid_only else np.arange(center.size)) if flat_indices is None else flat_indices
    arrays, names = [center.reshape(-1)[indices, None]], ['frozen_H_only_center']
    manifest = metadata.get('feature_manifest', {})
    included = {'H'}
    for key in sorted(key for key in raw if key.startswith('feature__')):
        name = key[9:]
        record = manifest.get(name, {})
        require(record.get('source') in ALLOWED_SOURCES and record.get('available_at_cutoff') is True,
                'Risk feature must be certified available at cutoff: '+name)
        channel = record.get('information_channel')
        require(channel in ('H', 'R', 'C'), 'Every risk feature needs an explicit H/R/C information channel')
        require((record['source'] == 'report') == (channel in ('R', 'C')),
                'Report-derived features cannot be relabeled H: '+name)
        if channel not in channels:
            continue
        value = np.asarray(raw[key])
        require(value.shape == checked['y'].shape or (value.ndim == 4 and value.shape[:3] == checked['y'].shape),
                'Risk feature axes disagree: '+name)
        require(value.dtype.kind in 'fiu', 'Risk feature must be numeric: '+name)
        if value.ndim == 3:
            value = value[..., None]
        selected = value.reshape(-1, value.shape[-1])[indices]
        require(np.isfinite(selected).all(), 'Risk feature must be finite: '+name)
        encoded, labels = encode_feature(selected, record, name)
        arrays.append(encoded)
        names.extend(labels)
        included.add(channel)
    require(channels <= included, 'Requested risk information is absent; do not label H features as a report comparison')
    require(len(names) <= 128, 'Risk head supports at most 128 documented features')
    x = np.concatenate(arrays, -1).astype(np.float64)
    require(np.isfinite(x).all(), 'Frozen center and risk features must be finite')
    return x, names


def _softplus(value):
    return np.logaddexp(0., value)


def _sigmoid(value):
    return np.exp(-np.logaddexp(0., -value))


def fit_risk_head(calibration_archive, validation_archive, calibration_metadata, validation_metadata,
                  mode='H', center_case='H_ONLY', candidate_case='BOTH', steps=300, batch_size=4096, seed=2025):
    """Fit only calibration y; fixed optimizer/budget, no validation selection."""
    require(isinstance(steps, int) and steps > 0 and isinstance(batch_size, int) and batch_size > 0,
            'Risk fitting needs a positive fixed step and batch budget')
    validate_archive_digest(calibration_archive, calibration_metadata)
    validate_archive_digest(validation_archive, validation_metadata)
    raw_c, raw_v = _mapping(calibration_archive), _mapping(validation_archive)
    cal, val = validate_archive(raw_c), validate_archive(raw_v)
    require(center_case == 'H_ONLY', 'Risk comparison must share the same frozen H_ONLY center')
    identity = validate_holdout_identity(cal, val, calibration_metadata, validation_metadata, center_case, candidate_case)
    require(calibration_metadata.get('feature_manifest', {}) == validation_metadata.get('feature_manifest', {}),
            'Risk feature definitions must be frozen across splits')
    eligible = np.flatnonzero(cal['valid'])
    fit_indices = eligible if len(eligible) <= 200000 else np.sort(np.random.default_rng(seed).choice(eligible, 200000, replace=False))
    x, names = _features(raw_c, cal, calibration_metadata, center_case, mode, True, fit_indices)
    _, val_names = _features(raw_v, val, validation_metadata, center_case, mode, True, np.array([0]))
    require(names == val_names, 'Risk feature axes differ across splits')
    require(len(x) > 1, 'Risk head needs at least two calibration points')
    center = cal['predictions'][center_case].reshape(-1)[fit_indices].astype(np.float64)
    residual = cal['y'].reshape(-1)[fit_indices].astype(np.float64)-center
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.
    x = (x-mean)/scale
    # Shared positive normalizer is fit on calibration only, for stable updates.
    target_scale = max(float(np.median(np.abs(residual))), 1e-3)
    residual = residual/target_scale
    initial = np.array([max(-float(np.quantile(residual, .1)), 1e-3),
                        max(float(np.quantile(residual, .9)), 1e-3)])
    bias = initial+np.log(-np.expm1(-initial))  # stable inverse softplus
    weight = np.zeros((x.shape[1], 2), np.float64)
    first, second = np.zeros_like(weight), np.zeros_like(weight)
    first_b, second_b = np.zeros(2), np.zeros(2)
    rng = np.random.default_rng(seed)
    # Fixed Adam settings and fixed budget; no validation labels or threshold grid.
    for step in range(1, steps+1):
        idx = rng.integers(len(x), size=min(batch_size, len(x)))
        batch, target = x[idx], residual[idx]
        logits = batch @ weight+bias
        half = _softplus(logits)
        q = np.column_stack((-half[:, 0], half[:, 1]))
        gradient_q = (q > target[:, None]).astype(np.float64)-np.array([.1, .9])
        gradient = gradient_q*_sigmoid(logits)*np.array([-1., 1.])
        g = batch.T @ gradient/len(idx)+1e-4*weight
        gb = gradient.mean(0)
        first, second = .9*first+.1*g, .999*second+.001*g*g
        first_b, second_b = .9*first_b+.1*gb, .999*second_b+.001*gb*gb
        weight -= .01*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
        bias -= .01*(first_b/(1-.9**step))/(np.sqrt(second_b/(1-.999**step))+1e-8)
    require(np.isfinite(weight).all() and np.isfinite(bias).all(), 'Risk head fit produced nonfinite parameters')
    return dict(schema=SCHEMA, mode=mode, center_case=center_case, feature_names=names,
                feature_manifest=calibration_metadata.get('feature_manifest', {}),
                mean=mean.tolist(), scale=scale.tolist(), weight=weight.tolist(), bias=bias.tolist(),
                residual_scale=target_scale, fit_points=len(x), eligible_points=len(eligible),
                point_subsample_cap=200000, steps=steps, seed=seed,
                batch_size=batch_size, optimizer='fixed_adam_0.01', quantile_levels=[.1, .5, .9],
                center_checkpoint_sha256=calibration_metadata['base_checkpoint_sha256'],
                candidate_checkpoint_sha256=calibration_metadata['candidate_checkpoint_sha256'],
                prediction_case_roles=validate_prediction_roles(calibration_metadata, center_case, candidate_case),
                fit_split='calibration', validation_labels_used_for_fitting=False,
                identity=identity, test_evaluated=False,
                caveat='positive_half_width_constraint_around_frozen_center_may_limit_true_conditional_quantiles')


def predict_risk_head(model, archive, metadata):
    require(model.get('schema') == SCHEMA, 'Unsupported risk head schema')
    validate_archive_digest(archive, metadata)
    require(validate_prediction_roles(metadata, model['center_case'], 'BOTH') == model['prediction_case_roles'],
            'Concrete prediction case roles changed after risk fitting')
    require(metadata.get('base_checkpoint_sha256') == model['center_checkpoint_sha256']
            and metadata.get('candidate_checkpoint_sha256') == model['candidate_checkpoint_sha256'],
            'Risk head must be refitted after source checkpoints change')
    require(metadata.get('feature_manifest', {}) == model['feature_manifest'], 'Risk feature contract changed')
    raw = _mapping(archive)
    checked = validate_archive(raw)
    center = checked['predictions'][model['center_case']].astype(np.float64)
    quantile = np.empty((center.size, 3), np.float64)
    for start in range(0, len(quantile), 65536):
        indices = np.arange(start, min(start+65536, len(quantile)))
        x, names = _features(raw, checked, metadata, model['center_case'], model['mode'], False, indices)
        require(names == model['feature_names'], 'Risk feature axis changed')
        half = _softplus(((x-np.asarray(model['mean']))/np.asarray(model['scale'])) @ np.asarray(model['weight'])
                         +np.asarray(model['bias']))*model['residual_scale']
        c = center.reshape(-1)[indices]
        quantile[indices] = np.column_stack((c-half[:, 0], c, c+half[:, 1]))
    quantile = quantile.reshape((*center.shape, 3))
    require(np.isfinite(quantile).all(), 'Risk head produced nonfinite intervals')
    return quantile


def evaluate_risk_heads(models, archive, metadata):
    checked = validate_archive(archive)
    return {name: quantile_metrics(predict_risk_head(model, archive, metadata), checked['y'], checked['valid'])
            for name, model in models.items()}


def run_risk_analysis(calibration_archive, validation_archive, calibration_metadata, validation_metadata, output_dir):
    output = Path(output_dir)
    require(not output.exists() or not any(output.iterdir()), 'Risk output must be new or empty')
    # All three use identical seed, fitting points, budget, and frozen H-only center.
    models, runtime = {}, {}
    for mode in MODES:
        model, timing = timed_cpu_call(fit_risk_head, calibration_archive, validation_archive,
                                      calibration_metadata, validation_metadata, mode)
        models[mode] = model
        runtime[mode] = dict(fit=dict(timing, processed_points=model['fit_points'], iterations=model['steps'],
                                      minibatch_point_draws=model['steps']*min(model['batch_size'], model['fit_points'])))
    raw = _mapping(validation_archive)
    checked = validate_archive(raw)
    quantiles = {}
    for mode, model in models.items():
        q, timing = timed_cpu_call(predict_risk_head, model, raw, validation_metadata)
        quantiles[mode] = q
        runtime[mode]['predict'] = dict(timing, processed_points=int(checked['y'].size),
                                       iterations=math.ceil(checked['y'].size/65536))
    scores = {mode: quantile_metrics(q, checked['y'], checked['valid']) for mode, q in quantiles.items()}
    report = dict(schema=SCHEMA, status='RISK_HEAD_DESCRIPTIVE_VALIDATION_COMPLETE', models=models, scores=scores,
                  center_case='H_ONLY', same_center_for_all_information_conditions=True, runtime=runtime,
                  deployment_required_source_model_forwards=1, complete_deployment_latency_measured=False,
                  timing_note='Conditional widths require the H_ONLY forecast and cutoff features; auxiliary CPU timings exclude that model forward',
                  validation_labels_used_for_fitting=False, test_evaluated=False,
                  boundaries=['three_quantiles_not_CRPS_or_full_distribution', 'no_conditional_coverage_guarantee',
                              'descriptive_holdout_not_independent_events',
                              'feature_channels_are_producer_assertions_not_independent_information_certificates',
                              'unequal_active_channels_not_equal_active_parameter_count'])
    output.mkdir(parents=True, exist_ok=True)
    archive = {key: value for key, value in raw.items() if not key.startswith('pred__') and not key.startswith('q__')}
    archive['pred__ORIGINAL'] = checked['predictions']['ORIGINAL']
    archive['quantile_levels'] = np.array([.1, .5, .9])
    for mode, q in quantiles.items():
        archive['pred__RISK_'+mode] = checked['predictions']['H_ONLY']
        archive['q__RISK_'+mode] = q
    np.savez_compressed(output/'risk_predictions.npz', **archive)
    analyze_archive(output/'risk_predictions.npz', output/'analysis', metadata=validation_metadata)
    report['paired_analysis_directory'] = 'analysis'
    (output/'risk.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-archive', required=True, type=Path)
    parser.add_argument('--validation-archive', required=True, type=Path)
    parser.add_argument('--calibration-metadata', required=True, type=Path)
    parser.add_argument('--validation-metadata', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_risk_analysis(args.calibration_archive, args.validation_archive,
                              json.loads(args.calibration_metadata.read_text(encoding='utf-8')),
                              json.loads(args.validation_metadata.read_text(encoding='utf-8')), args.output_dir)
    print(json.dumps(dict(schema=SCHEMA, status=report['status'], validation_fit=False, test_evaluated=False), indent=2))


if __name__ == '__main__':
    main()
