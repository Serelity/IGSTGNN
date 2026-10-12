"""Read-only, NumPy-only evaluation of saved window absolute-error totals.

These artifacts contain sums of absolute errors, not point predictions.  Window
statistics are descriptive: overlapping windows and repeated cutoffs are not
independent experimental repetitions.  A positive paired gain always means
that the candidate has a lower error than the control.
"""

from hashlib import sha256 as _sha256
from io import BytesIO
import math
from pathlib import Path

import numpy as np


HORIZONS = 12
SCHEMA = "saved_window_evaluation_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integers(value, name):
    array = np.asarray(value)
    _require(array.dtype.kind in "iuf", name + " must contain integers")
    if array.dtype.kind == "f":
        _require(bool(np.isfinite(array).all()), name + " must be finite")
        _require(bool((array == np.floor(array)).all()), name + " must contain integers")
        # int64.max rounds up when converted to float64; use an exclusive upper bound.
        _require(bool(((array >= -(2**63)) & (array < 2**63)).all()), name + " exceeds int64")
    elif array.dtype.kind == "u":
        _require(not array.size or int(array.max()) <= np.iinfo(np.int64).max, name + " exceeds int64")
    return array.astype(np.int64, copy=True)


def _count_totals(counts):
    # Python integer accumulation avoids a silent int64 overflow.
    totals = [sum(map(int, counts[:, h])) for h in range(HORIZONS)]
    _require(all(0 < value <= np.iinfo(np.int64).max for value in totals),
             "Every horizon must have a positive, representable total valid count")
    return np.asarray(totals, dtype=np.int64)


def _absolute_totals(absolute):
    try:
        totals = np.asarray([math.fsum(map(float, absolute[:, h])) for h in range(HORIZONS)], dtype=np.float64)
    except OverflowError as error:
        raise ValueError("Absolute-error totals exceed finite float64") from error
    _require(bool(np.isfinite(totals).all()), "Absolute-error totals must be finite")
    return totals


def _arrays(windows):
    required = ("sample_indices", "t0", "absolute", "counts")
    _require(all(key in windows for key in required), "Missing required saved-window fields")
    indices = _integers(windows["sample_indices"], "sample_indices")
    _require(indices.ndim == 1 and len(indices) > 0, "sample_indices must be a nonempty vector")
    _require(len(np.unique(indices)) == len(indices), "Duplicate sample_index in saved windows")
    raw_t0 = np.asarray(windows["t0"])
    _require(raw_t0.ndim == 1 and len(raw_t0) == len(indices), "t0 axes disagree with sample_indices")
    _require(raw_t0.dtype.kind in "US", "t0 must contain strings without object arrays")
    try:
        t0 = (np.char.decode(raw_t0, "utf-8") if raw_t0.dtype.kind == "S" else raw_t0).astype(str, copy=True)
    except UnicodeDecodeError as error:
        raise ValueError("t0 contains invalid UTF-8") from error
    _require(bool(np.char.str_len(t0).all()), "t0 contains an empty cutoff string")
    raw_absolute = np.asarray(windows["absolute"])
    _require(raw_absolute.dtype.kind in "iuf", "absolute must be a real numeric array")
    absolute = raw_absolute.astype(np.float64, copy=True)
    counts = _integers(windows["counts"], "counts")
    expected = (len(indices), HORIZONS)
    _require(absolute.shape == counts.shape == expected, "Expected [window,12] absolute and counts axes")
    _require(bool(np.isfinite(absolute).all()) and bool((absolute >= 0).all()),
             "Absolute errors must be finite and nonnegative")
    _require(bool((counts >= 0).all()), "Valid counts must be nonnegative")
    _require(not bool(((counts == 0) & (absolute != 0)).any()),
             "Zero valid count must have exactly zero absolute-error sum")
    _count_totals(counts)
    _absolute_totals(absolute)
    return dict(sample_indices=indices, t0=t0, absolute=absolute, counts=counts)


def validate_windows(path, expected_rows=None):
    """Load and validate one NPZ snapshot without pickle or file mutations.

    ``expected_rows`` is an optional positive integer row count.  The returned
    arrays are owned copies; ``sha256`` covers the exact byte snapshot loaded.
    """
    if expected_rows is not None:
        _require(isinstance(expected_rows, (int, np.integer)) and not isinstance(expected_rows, (bool, np.bool_))
                 and int(expected_rows) > 0, "expected_rows must be a positive integer")
    source = Path(path).resolve()
    snapshot = source.read_bytes()
    try:
        with np.load(BytesIO(snapshot), allow_pickle=False) as archive:
            normalized = _arrays({key: archive[key] for key in ("sample_indices", "t0", "absolute", "counts")})
    except KeyError as error:
        raise ValueError("Missing required saved-window field: " + str(error)) from error
    if expected_rows is not None:
        _require(len(normalized["sample_indices"]) == int(expected_rows), "Saved-window row count differs from expected_rows")
    normalized.update(path=str(source), sha256=_sha256(snapshot).hexdigest())
    return normalized


def _window_values(windows):
    available = (windows["counts"] > 0).all(axis=1)
    values = np.full(len(available), np.nan, dtype=np.float64)
    values[available] = (windows["absolute"][available] / windows["counts"][available] / HORIZONS).sum(axis=1)
    return available, values


def _mean(values):
    """Average finite values without overflowing a representable mean."""
    return math.fsum(float(value)/len(values) for value in values)


def _quantile(values, fraction):
    ordered = np.sort(values)
    position = (len(ordered)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position-lower
    # Convex interpolation avoids overflow from (upper - lower) for signed gains.
    return float((1-weight)*ordered[lower]+weight*ordered[upper])


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(mean=None, median=None, p05=None, p90=None, p95=None, minimum=None, maximum=None)
    return dict(mean=_mean(values), median=_quantile(values, .5),
                p05=_quantile(values, .05), p90=_quantile(values, .90),
                p95=_quantile(values, .95), minimum=float(values.min()), maximum=float(values.max()))


def _worst_indices(values, available, sample_indices, fraction):
    positions = np.flatnonzero(available)
    count = math.ceil(fraction * len(positions))
    # A fixed index tie-break makes equal-error tails reproducible after reordering.
    return np.asarray(sorted(map(int, positions), key=lambda i: (-float(values[i]), int(sample_indices[i])))[:count],
                      dtype=np.int64)


def window_summary(windows):
    """Return JSON-safe global metrics and descriptive complete-window tails."""
    normalized = _arrays(windows)
    total_counts = _count_totals(normalized["counts"])
    horizon_mae = _absolute_totals(normalized["absolute"]) / total_counts
    available, values = _window_values(normalized)
    tail = _worst_indices(values, available, normalized["sample_indices"], .05)
    unique_t0, frequencies = np.unique(normalized["t0"], return_counts=True)
    distribution = _distribution(values[available])
    distribution.update(available_count=int(available.sum()), excluded_count=int((~available).sum()),
                        partial_count=int(((normalized['counts'] > 0).any(axis=1) & ~available).sum()),
                        empty_count=int((normalized['counts'] == 0).all(axis=1).sum()),
                        definition="Equal mean of 12 per-horizon window MAEs; all 12 counts must be positive",
                        aggregation_warning="Equal-weight window mean need not equal the global horizon-macro MAE",
                        top_5_percent=dict(count=len(tail), mean=_mean(values[tail]) if len(tail) else None,
                                           sample_indices=normalized["sample_indices"][tail].tolist(),
                                           selection="ceil(0.05 * available_count); descending window MAE, sample_index ties"))
    return dict(schema=SCHEMA, sample_count=len(normalized["sample_indices"]), horizon_count=HORIZONS,
                mae_macro=_mean(horizon_mae), per_horizon_mae=horizon_mae.tolist(),
                valid_count_per_horizon=total_counts.tolist(), window_mae=distribution,
                time_dependence=dict(unique_t0_count=len(unique_t0),
                                     duplicate_t0_rows=len(normalized["t0"])-len(unique_t0),
                                     repeated_t0_count=int((frequencies > 1).sum()),
                                     maximum_windows_per_t0=int(frequencies.max()), independence_claim=False,
                                     note="Repeated cutoffs and overlapping windows are not IID repetitions; unique t0 is not an event count"),
                source_sha256=windows.get("sha256"), prediction_level="window_horizon_absolute_error_totals_only")


def compare_windows(control, candidate, tie_atol=1e-10):
    """Return ``(JSON-safe report, CSV-ready rows)`` after sample-index pairing.

    Window outcomes use the equal-horizon window mean.  Additive global
    contributions instead use the complete control population's horizon counts,
    so their sum exactly reconstructs the selected global metric's gain.  These
    are different denominators and must not be interpreted interchangeably.
    """
    _require(isinstance(tie_atol, (int, float, np.integer, np.floating))
             and not isinstance(tie_atol, (bool, np.bool_)) and math.isfinite(float(tie_atol))
             and float(tie_atol) >= 0, "tie_atol must be finite and nonnegative")
    left, right = _arrays(control), _arrays(candidate)
    _require(len(left["sample_indices"]) == len(right["sample_indices"]), "Paired sample_index sets differ")
    lookup = {int(index): i for i, index in enumerate(right["sample_indices"])}
    _require(all(int(index) in lookup for index in left["sample_indices"]), "Paired sample_index sets differ")
    order = np.asarray([lookup[int(index)] for index in left["sample_indices"]], dtype=np.int64)
    right = {key: value[order] for key, value in right.items()}
    _require(np.array_equal(left["t0"], right["t0"]), "Paired t0 differs after sample_index alignment")
    _require(np.array_equal(left["counts"], right["counts"]), "Paired valid counts differ after sample_index alignment")
    counts = _count_totals(left["counts"])
    control_h = _absolute_totals(left["absolute"]) / counts
    candidate_h = _absolute_totals(right["absolute"]) / counts
    difference = left["absolute"] - right["absolute"]
    horizon_gain = np.asarray([math.fsum(map(float, difference[:, h])) for h in range(HORIZONS)]) / counts
    main_gain = _mean(horizon_gain)
    contribution = (difference / counts[None, :] / HORIZONS).sum(axis=1)
    contribution_sum = math.fsum(map(float, contribution))
    additivity_error = contribution_sum - main_gain
    tolerance = 1e-10 * max(1., abs(main_gain), math.fsum(map(float, np.abs(contribution))))
    _require(abs(additivity_error) <= tolerance, "Window contributions do not reconstruct the global MAE gain")
    _require(abs(_mean(control_h)-_mean(candidate_h)-main_gain) <= tolerance,
             "Paired error differences do not reconstruct control minus candidate MAE")
    available, control_values = _window_values(left)
    _, candidate_values = _window_values(right)
    gains = control_values-candidate_values
    selected = gains[available]
    wins = int((selected > tie_atol).sum())
    losses = int((selected < -tie_atol).sum())
    ties = len(selected)-wins-losses
    tail = _worst_indices(control_values, available, left["sample_indices"], .10)
    tail_members = set(map(int, tail))
    outcomes = dict(available_count=len(selected), excluded_count=int((~available).sum()),
                    partial_count=int(((left['counts'] > 0).any(axis=1) & ~available).sum()),
                    empty_count=int((left['counts'] == 0).all(axis=1).sum()),
                    wins=wins, ties=ties, losses=losses,
                    win_fraction=wins/len(selected) if len(selected) else None,
                    tie_fraction=ties/len(selected) if len(selected) else None,
                    loss_fraction=losses/len(selected) if len(selected) else None, tie_atol=float(tie_atol))
    fixed_tail = dict(count=len(tail), sample_indices=left["sample_indices"][tail].tolist(),
                      control_window_mae_mean=_mean(control_values[tail]) if len(tail) else None,
                      candidate_window_mae_mean=_mean(candidate_values[tail]) if len(tail) else None,
                      window_mae_gain_mean=_mean(gains[tail]) if len(tail) else None,
                      selection="Control only; ceil(0.10 * available_count); descending control window MAE, sample_index ties",
                      global_mae_gain_contribution_sum=math.fsum(map(float, contribution[tail])))
    rows = []
    for i, sample_index in enumerate(left["sample_indices"]):
        outcome = None
        if available[i]:
            outcome = "win" if gains[i] > tie_atol else "loss" if gains[i] < -tie_atol else "tie"
        rows.append(dict(sample_index=int(sample_index), t0=str(left["t0"][i]), available=bool(available[i]),
                         control_window_mae=float(control_values[i]) if available[i] else None,
                         candidate_window_mae=float(candidate_values[i]) if available[i] else None,
                         window_mae_gain=float(gains[i]) if available[i] else None,
                         global_mae_gain_contribution=float(contribution[i]), outcome=outcome,
                         in_control_worst_10_percent=i in tail_members))
    report = dict(schema=SCHEMA, status="PAIRED_WINDOWS_VALID", sample_count=len(rows),
                  gain_sign="control_minus_candidate; positive means lower candidate error",
                  mae_macro=dict(control=_mean(control_h), candidate=_mean(candidate_h), gain=main_gain),
                  per_horizon=dict(control_mae=control_h.tolist(), candidate_mae=candidate_h.tolist(),
                                   gain=horizon_gain.tolist(), valid_count=counts.tolist()),
                  window_outcomes=outcomes, window_gain_distribution=_distribution(selected),
                  fixed_control_worst_10_percent=fixed_tail,
                  global_contribution_sum=contribution_sum, additivity_error=float(additivity_error),
                  contribution_definition="g_i = mean_h((A_control[i,h] - A_candidate[i,h]) / sum_i counts[i,h])",
                  window_gain_definition="Equal mean of 12 within-window horizon MAE differences; complete windows only",
                  pointwise_gain_decomposition_available=False, independent_repetition_claim=False,
                  control_source_sha256=control.get("sha256"), candidate_source_sha256=candidate.get("sha256"))
    return report, rows
