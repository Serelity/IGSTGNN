"""Read-only, origin-history cross diagnostics for four frozen GW13 arms.

The only dependencies are NumPy, the standard library, and the inherited fixed
GW10 flow edges. No model is fitted, predictions are not routed or changed,
and future labels never enter a grouping key. Blocks are consecutive positions
in the supplied original score order, not certified calendar intervals.
"""
import numpy as np

from experiments.conditional_residual_gw10.median_probe import FLOW_EDGES


ARMS = ('OWN4', 'MAGNITUDE16', 'DIRECTION16', 'TRAJECTORY16')
CONTROLS = ARMS[:-1]
BLOCK_SIZE = 128
SHAPES = ('early_down_recent_down', 'early_down_recent_nondecreasing',
          'early_nondecreasing_recent_down', 'early_nondecreasing_recent_nondecreasing')
TURNS = ('none', 'early_2_4', 'middle_5_6', 'late_7_9')
FLOWS = ('lt_0', '0_to_5', '5_to_20', '20_to_50', '50_to_100',
         '100_to_200', '200_to_314', 'ge_314')
# Every leaf stores sufficient statistics, never individual points. The first
# four fields count all valid points, nonzero truths, exact constants and zero
# net directions. Each arm contributes absolute/signed/squared/percentage error
# sums; each paired control contributes gross help and gross harm sums.
_ARM_OFFSET = 4
_PAIR_OFFSET = _ARM_OFFSET + 4 * len(ARMS)
_FIELDS = _PAIR_OFFSET + 2 * len(CONTROLS)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate(predictions, data, phase):
    _require(isinstance(phase, str) and bool(phase.strip()), 'A nonempty phase is required')
    _require(isinstance(predictions, dict) and set(predictions) == set(ARMS),
             'Exactly OWN4, MAGNITUDE16, DIRECTION16 and TRAJECTORY16 are required')
    _require(isinstance(data, dict) and all(key in data for key in
             ('y', 'valid', 'sample_ids', 'features')), 'Missing scoring data')
    truth, valid = np.asarray(data['y']), np.asarray(data['valid'])
    _require(truth.ndim == 3 and min(truth.shape) > 0 and truth.dtype.kind in 'fiu'
             and np.isfinite(truth).all(), 'Finite real [sample,lead,node] truth is required')
    _require(valid.shape == truth.shape and valid.dtype == np.bool_,
             'A matching Boolean valid mask is required')
    cubes = {name: np.asarray(predictions[name]) for name in ARMS}
    _require(all(cube.shape == truth.shape and cube.dtype.kind in 'fiu'
                 and np.isfinite(cube).all() for cube in cubes.values()),
             'All four predictions must be finite matching real cubes')
    samples, _, nodes = truth.shape
    ids = np.asarray(data['sample_ids'])
    _require(ids.shape == (samples,) and ids.dtype.kind in 'iu' and (ids >= 0).all()
             and np.unique(ids).size == samples,
             'Unique nonnegative integer sample_ids in original score order are required')
    features = data['features']
    names = ('shape', 'turn_position', 'last_flow', 'mean_abs_change', 'signed_direction')
    _require(isinstance(features, dict) and all(key in features for key in names),
             'Five cached history features are required')
    values = {key: np.asarray(features[key]) for key in names}
    _require(all(value.shape == (samples, nodes) and value.dtype.kind in 'fiu'
                 and np.isfinite(value).all() for value in values.values()),
             'History features must be finite real [sample,node] arrays')
    shape, turn = values['shape'], values['turn_position']
    _require(shape.dtype.kind in 'iu' and ((shape >= 0) & (shape < 4)).all(),
             'shape codes must be integers in 0..3')
    _require(turn.dtype.kind in 'iu' and ((turn == -1) | ((turn >= 2) & (turn <= 9))).all(),
             'turn_position must be -1 or an integer in 2..9')
    _require((values['mean_abs_change'] >= 0).all()
             and (np.abs(values['signed_direction']) <= 1.000001).all(),
             'Invalid cached mean_abs_change or signed_direction')
    return cubes, truth, valid, ids, values


def _history_keys(features):
    """Return compact, mutually exclusive history-only keys in 0..127."""
    turn = features['turn_position']
    turn_code = np.zeros(turn.shape, dtype=np.int16)
    turn_code[(turn >= 2) & (turn <= 4)] = 1
    turn_code[(turn >= 5) & (turn <= 6)] = 2
    turn_code[turn >= 7] = 3
    flow = np.searchsorted(np.asarray(FLOW_EDGES[1:-1]), features['last_flow'], side='right')
    return ((features['shape'].astype(np.int16) * 4 + turn_code) * 8
            + flow.astype(np.int16))


def _leaf(cubes, truth, valid, keys, features, start, stop, lead):
    """One bounded block/lead pass; no full-cube cell masks are constructed."""
    keep = valid[start:stop, lead, :].ravel()
    result = np.zeros((128, _FIELDS), dtype=np.float64)
    if not keep.any():
        return result
    group = keys[start:stop].ravel()[keep]
    target = truth[start:stop, lead, :].ravel()[keep].astype(np.float64)
    nonzero = target != 0.
    denominator = np.abs(target)

    def put(index, weights=None):
        result[:, index] = np.bincount(group, weights=weights, minlength=128)

    put(0)
    put(1, nonzero)
    put(2, features['mean_abs_change'][start:stop].ravel()[keep] == 0.)
    put(3, features['signed_direction'][start:stop].ravel()[keep] == 0.)
    absolute_errors = []
    for index, arm in enumerate(ARMS):
        error = cubes[arm][start:stop, lead, :].ravel()[keep].astype(np.float64) - target
        absolute = np.abs(error)
        offset = _ARM_OFFSET + index * 4
        put(offset, absolute)
        put(offset + 1, error)
        put(offset + 2, error * error)
        percentage = np.divide(absolute, denominator, out=np.zeros_like(absolute), where=nonzero)
        put(offset + 3, percentage)
        absolute_errors.append(absolute)
    for index in range(len(CONTROLS)):
        gain = absolute_errors[index] - absolute_errors[-1]
        put(_PAIR_OFFSET + index * 2, np.maximum(gain, 0.))
        put(_PAIR_OFFSET + index * 2 + 1, np.maximum(-gain, 0.))
    return result


def _row(moments, total, phase):
    n, mape_n = int(moments[0]), int(moments[1])
    result = dict(phase=phase, n=n, valid_count=n,
        valid_point_fraction=n / total if total else None,
        mape_nonzero_truth_count=mape_n,
        exact_constant_history_fraction=float(moments[2] / n) if n else None,
        net_direction_zero_fraction=float(moments[3] / n) if n else None)
    for index, arm in enumerate(ARMS):
        offset = _ARM_OFFSET + index * 4
        result[arm + '_mae'] = float(moments[offset] / n) if n else None
        result[arm + '_signed_bias'] = float(moments[offset + 1] / n) if n else None
        result[arm + '_rmse'] = float(np.sqrt(moments[offset + 2] / n)) if n else None
        result[arm + '_mape_fraction'] = float(moments[offset + 3] / mape_n) if mape_n else None
    for index, arm in enumerate(CONTROLS):
        help_sum, harm_sum = moments[_PAIR_OFFSET + index * 2:_PAIR_OFFSET + index * 2 + 2]
        net_sum = help_sum - harm_sum
        movement_sum = help_sum + harm_sum
        control_mae = result[arm + '_mae']
        prefix = 'trajectory_vs_' + arm + '_'
        gain = float(net_sum / n) if n else None
        result[prefix + 'mae_gain'] = gain
        result[prefix + 'relative_gain'] = gain / control_mae if control_mae else None
        result[prefix + 'overall_contribution'] = float(net_sum / total) if total and n else None
        result[prefix + 'gross_help'] = float(help_sum / n) if n else None
        result[prefix + 'gross_harm'] = float(harm_sum / n) if n else None
        result[prefix + 'cancellation'] = float(1. - abs(net_sum) / movement_sum) if movement_sum else None
    return result


def summarize(predictions, data, *, phase):
    """Describe frozen predictions using two alternative exhaustive partitions.

    Positive gain means lower TRAJECTORY16 absolute error. For a control,
    ``d = abs(control-y)-abs(TRAJECTORY16-y)`` on the *same* valid points:
    gross_help=mean(max(d,0)), gross_harm=mean(max(-d,0)), and
    cancellation=1-abs(mean(d))/(gross_help+gross_harm). These measure error
    magnitudes, not the fraction of helped/harmed samples. Contribution is
    sum(d in cell)/overall_valid_count; it is additive within one partition.

    Exact constants use only cached mean_abs_change==0; no plateau threshold
    is searched. The shape code is about two block-mean trends and does not
    establish strict monotonicity. Turn ``none`` means no strict adjacent
    smoothed-slope reversal, not a monotone or constant history.
    """
    cubes, truth, valid, ids, features = _validate(predictions, data, phase)
    samples, horizon, _ = truth.shape
    keys = _history_keys(features)
    blocks = (samples + BLOCK_SIZE - 1) // BLOCK_SIZE
    # About 10 MiB at production size: 31*12*128*26 float64 values. This
    # replaces repeated full prediction scans for each reported cell.
    leaves = np.zeros((blocks, horizon, 128, _FIELDS), dtype=np.float64)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for block, start in enumerate(range(0, samples, BLOCK_SIZE)):
                stop = min(start + BLOCK_SIZE, samples)
                for lead in range(horizon):
                    leaves[block, lead] = _leaf(cubes, truth, valid, keys, features, start, stop, lead)
            cross = leaves.sum(axis=(0, 1))
            phase_lead = leaves.sum(axis=0).reshape(horizon, 16, 8, _FIELDS).sum(axis=2)
            block_values = leaves.sum(axis=(1, 2))
            total_values = cross.sum(axis=0)
    except FloatingPointError as error:
        raise ValueError('Nonfinite diagnostic reduction from finite inputs') from error
    _require(np.isfinite(leaves).all() and np.isfinite(total_values).all(),
             'Nonfinite diagnostic reduction from finite inputs')
    total = int(total_values[0])
    cross_rows, phase_lead_rows, block_rows = [], [], []
    for shape in range(4):
        for turn in range(4):
            dimensions = dict(shape_code=shape, shape=SHAPES[shape], turn_code=turn, turn=TURNS[turn])
            for flow in range(8):
                moments = cross[(shape * 4 + turn) * 8 + flow]
                if moments[0]:
                    cross_rows.append(dict(dimensions, flow_bin=flow, flow=FLOWS[flow],
                                           **_row(moments, total, phase)))
            for lead in range(horizon):
                moments = phase_lead[lead, shape * 4 + turn]
                if moments[0]:
                    phase_lead_rows.append(dict(dimensions, lead=lead + 1,
                                                **_row(moments, total, phase)))
    for block, start in enumerate(range(0, samples, BLOCK_SIZE)):
        stop = min(start + BLOCK_SIZE, samples)
        block_rows.append(dict(block_index=block, score_position_start=start,
            score_position_stop_exclusive=stop, sample_count=stop - start,
            sample_id_first=int(ids[start]), sample_id_last=int(ids[stop - 1]),
            **_row(block_values[block], total, phase)))
    overall = _row(total_values, total, phase)
    overall.update(sample_count=samples, horizon=horizon, node_count=truth.shape[2],
        cross_total_combinations=4 * 4 * 8, cross_observed_combinations=len(cross_rows),
        phase_lead_total_combinations=4 * 4 * horizon,
        phase_lead_observed_combinations=len(phase_lead_rows), block_count=blocks,
        empty_cross_cells='omitted_not_zero_gain', empty_phase_lead_cells='omitted_not_zero_gain',
        mape_definition='mean(abs(prediction-truth)/abs(truth)) over valid points with truth!=0',
        fraction_denominator='same overall valid prediction points for all four arms',
        gain_definition='mean(abs(control-truth)-abs(TRAJECTORY16-truth)); positive favors TRAJECTORY16',
        contribution_definition='sum paired pointwise gain in cell / overall valid point count',
        paired_magnitude_definition='gross_help and gross_harm are mean positive and negative gain magnitudes, not sample fractions',
        cancellation_definition='1-abs(net_gain)/(gross_help+gross_harm); null if no error movement',
        partitions='cross, phase_lead and block each partition the same valid points; do not add partitions together',
        exact_constant_history_definition='cached mean_abs_change==0, weighted by valid points; no threshold search',
        net_direction_zero_definition='cached signed_direction==0, weighted by valid points',
        shape_is_strict_monotonicity=False,
        turn_none_definition='no strict adjacent smoothed-slope reversal; does not imply monotonicity',
        flow_boundary_definition='inherited GW10 FLOW_EDGES; lower inclusive, upper exclusive; ties go right',
        flow_interior_edges=list(FLOW_EDGES[1:-1]),
        block_size=BLOCK_SIZE, block_definition='consecutive supplied score-order positions, preserving sample_ids',
        blocks_are_independent_calendar_intervals=False, grouping_uses_future_truth=False,
        predictions_rerouted=False, model_fits=0)
    return dict(cross_rows=cross_rows, phase_lead_rows=phase_lead_rows,
                block_rows=block_rows, overall=overall)
