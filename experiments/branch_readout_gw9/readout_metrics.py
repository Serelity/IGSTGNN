"""Descriptive paired-cache diagnostics with no fitting or prediction changes.

All arithmetic uses float64 on the selected valid points. Group contributions
use the caller's *global* valid denominator, while each group's ordinary MAE
uses its own count. Validation-label summaries are descriptive evidence only;
the returned means must not be applied as fitted prediction corrections.

This module depends on NumPy and the standard library only. It deliberately
does not import model, training, probe, or checkpoint-loading code.
"""
import math

import numpy as np


CASES = ('FROZEN_R', 'SUM_CONTROL', 'BRANCH_READOUT')
PAIRS = (
    ('SUM_vs_R', 'FROZEN_R', 'SUM_CONTROL'),
    ('BRANCH_vs_SUM', 'SUM_CONTROL', 'BRANCH_READOUT'),
    ('BRANCH_vs_R', 'FROZEN_R', 'BRANCH_READOUT'),
)


def _mean(values):
    return float(np.mean(values, dtype=np.float64)) if values.size else None


def _rms(values):
    if not values.size:
        return None
    return float(np.sqrt(np.dot(values, values) / values.size))


def _quantile(values, probability):
    return float(np.quantile(values, probability)) if values.size else None


def _correlation(left, right):
    """Pearson correlation; an empty or exactly constant operand is undefined."""
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    centered_left = left - np.mean(left, dtype=np.float64)
    centered_right = right - np.mean(right, dtype=np.float64)
    denominator = math.sqrt(float(np.dot(centered_left, centered_left)))
    denominator *= math.sqrt(float(np.dot(centered_right, centered_right)))
    if denominator == 0.0:
        return None
    value = float(np.dot(centered_left, centered_right)) / denominator
    return float(np.clip(value, -1.0, 1.0))


def _selection(y, valid, mask):
    y = np.asarray(y)
    valid = np.asarray(valid)
    if not np.issubdtype(y.dtype, np.number) or np.iscomplexobj(y):
        raise ValueError('Truth must be a real numeric array')
    if valid.dtype != np.bool_:
        raise ValueError('The validity mask must have Boolean dtype')
    try:
        selected = np.broadcast_to(valid, y.shape)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != np.bool_:
                raise ValueError('The group mask must have Boolean dtype')
            selected = selected & np.broadcast_to(mask, y.shape)
    except ValueError as error:
        raise ValueError('Masks must be Boolean and broadcast to truth shape') from error
    return y, selected


def _selected_float64(values, selected, shape, label):
    values = np.asarray(values)
    if values.shape != shape:
        raise ValueError(label + ' shape differs from truth')
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(label + ' must be a real numeric array')
    result = np.asarray(values[selected], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(label + ' contains nonfinite selected values')
    return result


def _case_metrics(prediction, truth):
    count = int(truth.size)
    error = prediction - truth
    absolute = np.abs(error)
    nonzero = truth != 0.0
    mape_count = int(np.count_nonzero(nonzero))
    return dict(
        count=count,
        valid_count=count,
        mae=_mean(absolute),
        rmse=_rms(error),
        mape_fraction=_mean(absolute[nonzero] / np.abs(truth[nonzero])),
        mape_valid_count=mape_count,
        signed_bias=_mean(error),
        under_fraction=float(np.count_nonzero(error < 0.0) / count) if count else None,
        over_fraction=float(np.count_nonzero(error > 0.0) / count) if count else None,
        exact_fraction=float(np.count_nonzero(error == 0.0) / count) if count else None,
        signed_error_p05=_quantile(error, .05),
        signed_error_p50=_quantile(error, .50),
        signed_error_p95=_quantile(error, .95),
        absolute_error_p50=_quantile(absolute, .50),
        absolute_error_p95=_quantile(absolute, .95),
    )


def _paired_metrics(before, after, truth, total_valid_count):
    """Positive point gain means the candidate reduced absolute error."""
    count = int(truth.size)
    before_error = before - truth
    after_error = after - truth
    before_absolute = np.abs(before_error)
    gain = before_absolute - np.abs(after_error)
    gain_sum = float(np.sum(gain, dtype=np.float64))
    help_sum = float(np.sum(gain[gain > 0.0], dtype=np.float64))
    harm_sum = float(-np.sum(gain[gain < 0.0], dtype=np.float64))
    movement = help_sum + harm_sum
    before_mae = _mean(before_absolute)
    mae_gain = gain_sum / count if count else None
    global_gain = gain_sum / total_valid_count if count else None
    cancellation = 1.0 - abs(gain_sum) / movement if movement else None
    # This is an identity for the actual paired corrections, not an estimated
    # offset applied to forecasts. Centering is used only for decomposition.
    correction = after - before
    before_bias = _mean(before_error)
    mean_correction = _mean(correction)
    before_mse = float(np.dot(before_error, before_error) / count) if count else None
    after_mse = float(np.dot(after_error, after_error) / count) if count else None
    mean_change = centered_change = total_change = closure_error = None
    if count:
        centered_error = before_error - before_bias
        centered_correction = correction - mean_correction
        covariance = float(np.dot(centered_error, centered_correction) / count)
        variance = float(np.dot(centered_correction, centered_correction) / count)
        mean_change = 2.0 * before_bias * mean_correction + mean_correction ** 2
        centered_change = 2.0 * covariance + variance
        total_change = after_mse - before_mse
        closure_error = total_change - (mean_change + centered_change)
        scale = max(1.0, abs(before_mse), abs(after_mse),
            abs(mean_change), abs(centered_change))
        if not math.isclose(total_change, mean_change + centered_change,
                rel_tol=1e-10, abs_tol=64.0 * np.finfo(np.float64).eps * scale):
            raise ValueError('Paired MSE decomposition did not close in float64')
    return dict(
        count=count,
        valid_count=count,
        total_valid_count=int(total_valid_count),
        mae_gain=mae_gain,
        relative_gain=mae_gain / before_mae if count and before_mae != 0.0 else None,
        global_mae_contribution=global_gain,
        improved_fraction=float(np.count_nonzero(gain > 0.0) / count) if count else None,
        worsened_fraction=float(np.count_nonzero(gain < 0.0) / count) if count else None,
        tied_fraction=float(np.count_nonzero(gain == 0.0) / count) if count else None,
        gross_help=help_sum / total_valid_count if count else None,
        gross_harm=harm_sum / total_valid_count if count else None,
        net=global_gain,
        cancellation_ratio=float(np.clip(cancellation, 0.0, 1.0))
            if cancellation is not None else None,
        group_gross_help=help_sum / count if count else None,
        group_gross_harm=harm_sum / count if count else None,
        mse_decomposition=dict(
            before_mse=before_mse,
            after_mse=after_mse,
            mse_gain=before_mse - after_mse if count else None,
            mean_correction=mean_correction,
            before_signed_bias=before_bias,
            mean_only_mse_change=mean_change,
            centered_mse_change=centered_change,
            total_mse_change=total_change,
            identity_closure_error=closure_error,
        ),
    )


def _correction_summary(correction, before_error):
    count = int(correction.size)
    mean = _mean(correction)
    return dict(
        count=count,
        mean=mean,
        rms=_rms(correction),
        std=float(np.std(correction, dtype=np.float64)) if count else None,
        abs_mean=_mean(np.abs(correction)),
        max_abs=float(np.max(np.abs(correction))) if count else None,
        p95_abs=_quantile(np.abs(correction), .95),
        correlation_with_before_signed_error=_correlation(correction, before_error),
    )


def analyze_group(predictions, y, valid, mask=None, total_valid_count=None):
    """Summarize the three paired cases on an explicitly masked group.

    ``mask`` and ``valid`` must be Boolean and broadcast to ``y.shape``.
    Predictions must have exactly that shape. Selected finite values are
    converted to float64 one case/pair at a time; no full float64 error tensor
    is retained across calls (in particular when callers iterate over nodes).

    MAPE excludes zero truth values and reports its separate denominator;
    all other metrics use precisely the supplied validity mask. Empty groups
    contain count zero and undefined numeric summaries represented by None.
    """
    y, selected = _selection(y, valid, mask)
    missing = [case for case in CASES if case not in predictions]
    if missing:
        raise ValueError('Missing prediction cases: ' + ', '.join(missing))
    truth = _selected_float64(y, selected, y.shape, 'Truth')
    count = int(truth.size)
    if total_valid_count is None:
        total_valid_count = count
    if (isinstance(total_valid_count, (bool, np.bool_))
            or not isinstance(total_valid_count, (int, np.integer))
            or total_valid_count < count):
        raise ValueError('Global valid count must be an integer at least the group count')

    cases = {}
    for case in CASES:
        values = _selected_float64(predictions[case], selected, y.shape, case)
        cases[case] = _case_metrics(values, truth)
        del values

    comparisons = {}
    for name, before_case, after_case in PAIRS:
        before = _selected_float64(predictions[before_case], selected, y.shape, before_case)
        after = _selected_float64(predictions[after_case], selected, y.shape, after_case)
        comparisons[name] = _paired_metrics(before, after, truth, total_valid_count)
        del before, after

    baseline = _selected_float64(predictions['FROZEN_R'], selected, y.shape, 'FROZEN_R')
    sum_correction = _selected_float64(
        predictions['SUM_CONTROL'], selected, y.shape, 'SUM_CONTROL') - baseline
    branch_correction = _selected_float64(
        predictions['BRANCH_READOUT'], selected, y.shape, 'BRANCH_READOUT') - baseline
    before_error = baseline - truth
    del baseline
    corrections = {
        'SUM_minus_R':_correction_summary(sum_correction, before_error),
        'BRANCH_minus_R':_correction_summary(branch_correction, before_error),
        'BRANCH_minus_SUM':_correction_summary(
            branch_correction - sum_correction, before_error + sum_correction),
        'common_correction_correlation':_correlation(sum_correction, branch_correction),
    }
    delta_rms = corrections['BRANCH_minus_SUM']['rms']
    sum_rms = corrections['SUM_minus_R']['rms']
    branch_rms = corrections['BRANCH_minus_R']['rms']
    corrections['branch_delta_to_sum_rms_ratio'] = (
        delta_rms / sum_rms if count and sum_rms != 0.0 else None)
    corrections['branch_delta_to_branch_rms_ratio'] = (
        delta_rms / branch_rms if count and branch_rms != 0.0 else None)
    return dict(cases=cases, comparisons=comparisons, corrections=corrections)


def decompose_intercept(values):
    """Describe the constant component's energy without modifying predictions.

    The mean is computed on validation values for description only. This is
    not a fitted calibrator, a deployable correction, or an oracle score.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('Correction decomposition requires finite values')
    count = int(values.size)
    mean = _mean(values)
    energy = float(np.dot(values, values))
    constant_energy = count * mean * mean if count else 0.0
    share = float(np.clip(constant_energy / energy, 0.0, 1.0)) if energy else None
    return dict(
        count=count,
        mean=mean,
        rms=_rms(values),
        sum_squares=energy if count else None,
        global_constant_energy_share=share,
        centered_energy_share=1.0 - share if share is not None else None,
    )


def correction_decomposition(correction, valid):
    """Describe global and horizon constant energy on sample/horizon/node axes.

    The horizon share *includes* the global constant component. Thus the
    orthogonal decomposition is global + additional horizon + within-horizon
    variation. Missing horizons have zero count and undefined local means.
    These are validation summaries, not predictions with estimated offsets.
    """
    correction, selected = _selection(correction, valid, None)
    if correction.ndim != 3:
        raise ValueError('Correction axes must be sample, horizon, node')
    global_values = _selected_float64(correction, selected, correction.shape, 'Correction')
    result = decompose_intercept(global_values)
    energy = result['sum_squares']
    horizons = []
    horizon_energy = 0.0
    for index in range(correction.shape[1]):
        local = np.asarray(correction[:, index, :][selected[:, index, :]], dtype=np.float64)
        record = decompose_intercept(local)
        record['horizon'] = index + 1
        horizons.append(record)
        if record['count']:
            horizon_energy += record['count'] * record['mean'] * record['mean']
    horizon_share = float(np.clip(horizon_energy / energy, 0.0, 1.0)) if energy else None
    global_share = result['global_constant_energy_share']
    result.update(
        horizons=horizons,
        horizon_constant_energy_share=horizon_share,
        additional_horizon_energy_share=max(0.0, horizon_share - global_share)
            if horizon_share is not None else None,
        within_horizon_centered_energy_share=1.0 - horizon_share
            if horizon_share is not None else None,
    )
    return result
