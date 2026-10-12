"""NumPy-only descriptive metrics for a same-checkpoint TIID x D intervention.

The four cases share labels, valid masks, sample order and model weights. These
metrics describe checkpoint dependence; they do not estimate independent event
effects or the quality of a separately retrained, pruned architecture.
"""

import math

import numpy as np


SCHEMA = "b3r_module_factorial_metrics_v1"
CASES = ("TIID_ON_D_ON", "TIID_ON_D_OFF", "TIID_OFF_D_ON", "TIID_OFF_D_OFF")
COMPARISONS = {
    "D_given_TIID_ON": (CASES[1], CASES[0]),
    "D_given_TIID_OFF": (CASES[3], CASES[2]),
    "TIID_given_D_ON": (CASES[2], CASES[0]),
    "TIID_given_D_OFF": (CASES[3], CASES[1]),
    "combined_vs_both_off": (CASES[3], CASES[0]),
    "TIID_only_vs_D_only": (CASES[2], CASES[1]),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _numeric(value, name):
    array = np.asarray(value)
    _require(array.dtype.kind in "iuf", name + " must be a real numeric array")
    return array.astype(np.float64, copy=False)


def _mean(values):
    return math.fsum(float(value) / len(values) for value in values) if len(values) else None


def _per_horizon(sums, counts):
    return [float(value / count) if count else None for value, count in zip(sums, counts)]


def _macro(sums, counts):
    return _mean(sums / counts) if bool((counts > 0).all()) else None


def _rank_tail(values, available, indices, fraction):
    positions = np.flatnonzero(available)
    number = math.ceil(fraction * len(positions))
    return np.asarray(sorted(map(int, positions), key=lambda i: (-float(values[i]), int(indices[i])))[:number],
                      dtype=np.int64)


def _window_values(absolute, counts):
    available = (counts > 0).all(axis=1)
    values = np.full(len(counts), np.nan, dtype=np.float64)
    values[available] = (absolute[available] / counts[available]).mean(axis=1)
    return available, values


def _window_summary(absolute, counts, indices):
    available, values = _window_values(absolute, counts)
    selected = values[available]
    worst = _rank_tail(values, available, indices, .05)
    return dict(
        available_count=int(available.sum()), excluded_count=int((~available).sum()),
        partial_count=int(((counts > 0).any(axis=1) & ~available).sum()),
        empty_count=int((counts == 0).all(axis=1).sum()),
        mean=_mean(selected), median=float(np.median(selected)) if len(selected) else None,
        p95=float(np.quantile(selected, .95)) if len(selected) else None,
        top_5_percent=dict(count=len(worst), mean=_mean(values[worst]), sample_indices=indices[worst].tolist()),
        definition="Equal mean of per-horizon window MAEs; every horizon must have a positive valid count",
        selection="Own-case worst ceil(0.05 * available_count); descending MAE, ascending sample_index ties",
        aggregation_warning="Equal-weight window mean need not equal horizon-macro population MAE",
    )


def _case_metric(raw, indices):
    absolute, squared, counts = (raw[name] for name in ("absolute", "squared", "counts"))
    totals = counts.sum(axis=0)
    absolute_total, squared_total = absolute.sum(axis=0), squared.sum(axis=0)
    horizon_rmse = [float(math.sqrt(value / count)) if count else None
                    for value, count in zip(squared_total, totals)]
    available = bool((totals > 0).all())
    return dict(status="available" if available else "unavailable", valid_count=int(totals.sum()),
                valid_count_per_horizon=totals.tolist(),
                per_horizon_mae=_per_horizon(absolute_total, totals),
                per_horizon_rmse=horizon_rmse,
                mae_macro=_macro(absolute_total, totals),
                rmse_macro=_mean(horizon_rmse) if available else None,
                mae_pooled=float(absolute_total.sum() / totals.sum()) if totals.sum() else None,
                rmse_pooled=float(math.sqrt(squared_total.sum() / totals.sum())) if totals.sum() else None,
                window_mae=_window_summary(absolute, counts, indices))


def _gain_metric(gain, baseline_prediction, candidate_prediction, selected, global_counts, raw_baseline, raw_candidate):
    counts = selected.sum(axis=(0, 2))
    sums = np.where(selected, gain, 0.).sum(axis=(0, 2))
    positive = np.where(selected, np.maximum(gain, 0.), 0.).sum(axis=(0, 2))
    negative = np.where(selected, np.maximum(-gain, 0.), 0.).sum(axis=(0, 2))
    change = candidate_prediction - baseline_prediction
    absolute_change = np.where(selected, np.abs(change), 0.).sum(axis=(0, 2))
    signed_change = np.where(selected, change, 0.).sum(axis=(0, 2))
    net, good, bad = (_macro(value, counts) for value in (sums, positive, negative))
    gross = good + bad if good is not None else None
    global_net, global_good, global_bad = (_mean(value / global_counts) for value in (sums, positive, negative))
    available, baseline_windows = _window_values(raw_baseline["absolute"], raw_baseline["counts"])
    _, candidate_windows = _window_values(raw_candidate["absolute"], raw_candidate["counts"])
    window_gains = (baseline_windows - candidate_windows)[available]
    wins, losses = int((window_gains > 1e-10).sum()), int((window_gains < -1e-10).sum())
    ties = len(window_gains) - wins - losses
    return dict(
        status="available" if bool((counts > 0).all()) else "unavailable",
        valid_count=int(counts.sum()), valid_count_per_horizon=counts.tolist(),
        baseline_minus_candidate_mae=net, per_horizon_baseline_minus_candidate_mae=_per_horizon(sums, counts),
        gain_positive=good, gain_negative=bad, gross_error_gain=gross,
        cancellation_fraction=None if gross is None or gross == 0 else max(0., 1. - abs(net) / gross),
        per_horizon_gain_positive=_per_horizon(positive, counts),
        per_horizon_gain_negative=_per_horizon(negative, counts),
        global_gain_contribution=global_net, global_gain_positive=global_good, global_gain_negative=global_bad,
        global_valid_weight_fraction=_mean(counts / global_counts),
        global_cancellation_fraction=None if global_good + global_bad == 0 else
            max(0., 1. - abs(global_net) / (global_good + global_bad)),
        per_horizon_global_gain_contribution=(sums / global_counts).tolist(),
        prediction_absolute_change_macro=_macro(absolute_change, counts),
        prediction_signed_change_candidate_minus_baseline_macro=_macro(signed_change, counts),
        prediction_absolute_change_max=float(np.abs(change[selected]).max()) if selected.any() else None,
        window_outcomes=dict(available_count=len(window_gains), excluded_count=int((~available).sum()),
                             wins=wins, losses=losses, ties=ties,
                             win_fraction=wins / len(window_gains) if len(window_gains) else None,
                             baseline_minus_candidate_mae_mean=_mean(window_gains), tie_atol=1e-10,
                             independence_claim=False),
    )


def summarize_factorial(predictions, labels, regions, *, valid=None, sample_indices=None, fixed_hard_fraction=.1):
    """Summarize four paired ``[row, horizon, node]`` prediction arrays.

    ``valid`` must be boolean with the exact label shape. Zero-valued labels are
    valid unless explicitly masked. Targets may be nonfinite only where valid
    is false; predictions must always be finite. ``all_nodes`` must be present
    and cover every row/node. Region masks may overlap and are not additive.

    Interaction is M10 + M01 - M11 - M00, where subscripts are TIID,D switches.
    Positive interaction means the combined MAE gain exceeds the sum of the
    two single-path gains relative to both-off, for these fixed model weights.
    """
    _require(isinstance(predictions, dict) and set(predictions) == set(CASES), "Exactly the four TIID x D cases are required")
    labels = _numeric(labels, "labels")
    _require(labels.ndim == 3 and all(size > 0 for size in labels.shape), "Expected nonempty [row,horizon,node] labels")
    rows, horizons, nodes = labels.shape
    if valid is None:
        valid = np.ones(labels.shape, dtype=np.bool_)
    valid = np.asarray(valid)
    _require(valid.dtype == np.bool_ and valid.shape == labels.shape, "valid must be boolean [row,horizon,node] with exact label axes")
    _require(bool(np.isfinite(labels[valid]).all()), "Valid targets must be finite")
    global_counts = valid.sum(axis=(0, 2))
    _require(bool((global_counts > 0).all()), "Every global horizon needs a positive valid count")
    if sample_indices is None:
        sample_indices = np.arange(rows, dtype=np.int64)
    indices = np.asarray(sample_indices)
    _require(indices.shape == (rows,) and indices.dtype.kind in "iu" and len(np.unique(indices)) == rows,
             "sample_indices must be a unique integer row vector")
    _require(isinstance(fixed_hard_fraction, (int, float, np.integer, np.floating))
             and not isinstance(fixed_hard_fraction, (bool, np.bool_))
             and math.isfinite(float(fixed_hard_fraction)) and 0 < fixed_hard_fraction <= 1,
             "fixed_hard_fraction must be finite in (0,1]")
    _require(isinstance(regions, dict) and "all_nodes" in regions, "regions must contain all_nodes")
    masks = {}
    for name, value in regions.items():
        _require(isinstance(name, str) and bool(name), "Region names must be nonempty strings")
        mask = np.asarray(value)
        _require(mask.dtype == np.bool_ and mask.shape == (rows, nodes), "Region must be boolean [row,node]: " + name)
        masks[name] = mask
    _require(bool(masks["all_nodes"].all()), "all_nodes must cover the entire population")
    arrays, errors, raw = {}, {}, {}
    try:
        with np.errstate(over="raise", invalid="raise"):
            safe_labels = np.where(valid, labels, 0.)
            for case in CASES:
                prediction = _numeric(predictions[case], case)
                _require(prediction.shape == labels.shape, "Prediction axes differ: " + case)
                _require(bool(np.isfinite(prediction).all()), "Predictions must be finite: " + case)
                arrays[case] = prediction
                error = np.abs(np.where(valid, prediction, 0.) - safe_labels)
                errors[case] = error
                squared = np.square(error)
                raw[case] = {}
                for name, mask in masks.items():
                    selected = valid & mask[:, None, :]
                    raw[case][name] = dict(counts=selected.sum(axis=2),
                                           absolute=np.where(selected, error, 0.).sum(axis=2),
                                           squared=np.where(selected, squared, 0.).sum(axis=2))
            cases = {case: dict(metrics={name: _case_metric(value, indices) for name, value in raw[case].items()})
                     for case in CASES}
            comparisons = {}
            for name, (baseline, candidate) in COMPARISONS.items():
                gain = errors[baseline] - errors[candidate]
                comparisons[name] = dict(baseline=baseline, candidate=candidate, metrics={
                    region: _gain_metric(gain, arrays[baseline], arrays[candidate], valid & mask[:, None, :],
                                         global_counts, raw[baseline][region], raw[candidate][region])
                    for region, mask in masks.items()})
            interaction = {}
            for region in masks:
                metrics = [cases[case]["metrics"][region] for case in CASES]
                per_horizon = [None if metrics[0]["valid_count_per_horizon"][h] == 0 else
                               metrics[1]["per_horizon_mae"][h] + metrics[2]["per_horizon_mae"][h]
                               - metrics[0]["per_horizon_mae"][h] - metrics[3]["per_horizon_mae"][h]
                               for h in range(horizons)]
                available = all(value is not None for value in per_horizon)
                interaction[region] = dict(status="available" if available else "unavailable",
                                           per_horizon_mae_interaction=per_horizon,
                                           mae_interaction_macro=_mean(per_horizon) if available else None)
            on_raw = raw[CASES[0]]["all_nodes"]
            available, on_windows = _window_values(on_raw["absolute"], on_raw["counts"])
            hard = _rank_tail(on_windows, available, indices, float(fixed_hard_fraction))
            hard_cases = {}
            for case in CASES:
                hard_metrics = {}
                for region in masks:
                    subset = {key: value[hard] for key, value in raw[case][region].items()}
                    metric = _case_metric(subset, indices[hard])
                    reference = raw[CASES[0]][region]
                    reference_available, reference_windows = _window_values(reference["absolute"][hard], reference["counts"][hard])
                    _, candidate_windows = _window_values(subset["absolute"], subset["counts"])
                    metric["full_on_minus_case_window_mae_mean"] = _mean(
                        (reference_windows - candidate_windows)[reference_available])
                    hard_metrics[region] = metric
                hard_cases[case] = dict(metrics=hard_metrics)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Metrics exceed finite float64 arithmetic") from error
    return dict(schema=SCHEMA, case_order=list(CASES), sample_count=rows, horizon_count=horizons,
                node_count=nodes, gain_sign="baseline_minus_candidate; positive means candidate improves MAE",
                cases=cases, comparisons=comparisons,
                interaction=dict(definition="M10 + M01 - M11 - M00; positive means combined MAE gain exceeds single-path gains",
                                 metrics=interaction),
                fixed_hard_windows=dict(reference_case=CASES[0], reference_region="all_nodes",
                                        fraction=float(fixed_hard_fraction), count=len(hard),
                                        sample_indices=indices[hard].tolist(), row_positions=hard.tolist(),
                                        selection="Full-ON complete windows only; descending MAE, ascending sample_index ties",
                                        cases=hard_cases),
                interpretation=dict(scope="same_checkpoint_conditional_offline_sensitivity",
                                    independence_claim=False, significance_test=False,
                                    regions_may_overlap=True, zero_targets_automatically_masked=False,
                                    pruning_requires_retraining=True,
                                    warning="Overlapping windows are descriptive observations, not independent replications; OFF may be out of training distribution"))
