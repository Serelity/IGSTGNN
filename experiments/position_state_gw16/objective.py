"""One raw-unit objective and float64 pooled metrics for all GW16 arms."""
import numpy as np
import torch


def valid_target(target, config):
    return torch.isfinite(target) & (target.double() > float(config['target_minimum']))


def region_residual(prediction, target, geometry, config):
    valid = valid_target(target, config)
    coverage = geometry.pool(valid.to(prediction.dtype))
    residual = torch.where(valid, prediction-target, torch.zeros_like(prediction))
    error = geometry.pool(residual)/coverage.clamp_min(1e-12)
    return error, coverage >= config['minimum_region_coverage']


def loss_terms(output, target, geometry, config, stage='joint'):
    if stage != 'joint':
        raise ValueError('GW16 has only joint training')
    valid = valid_target(target, config)
    if not bool(valid.any()):
        raise ValueError('Batch has no valid raw-unit targets')
    prediction = output['prediction']
    node = (prediction-target).abs()[valid].mean()
    error, mask = region_residual(prediction, target, geometry, config)
    region = error.abs()[mask].mean() if bool(mask.any()) else prediction.sum()*0
    return dict(loss=node+config['region_loss_weight']*region,
                node_mae=node, region_mae=region)


class PooledStatistics:
    def __init__(self, config):
        self.config = config
        self.total = dict(count=0, absolute=0., square=0., signed=0., negative=0,
                          mape_count=0, percentage=0., region_count=0,
                          region_absolute=0., reference_absolute=0.)

    def update(self, output, target, geometry):
        p, b, y = output['prediction'].detach().double(), output['reference'].detach().double(), target.double()
        if not bool(torch.isfinite(p).all() and torch.isfinite(b).all()):
            raise ValueError('Nonfinite predictions or anchors')
        mask = valid_target(target, self.config)
        mm = mask & (y >= self.config['mape_minimum'])
        e = (p-y)[mask]
        t = self.total
        t['count'] += int(mask.sum())
        t['absolute'] += float(e.abs().sum())
        t['square'] += float(e.square().sum())
        t['signed'] += float(e.sum())
        t['negative'] += int((p[mask] < 0).sum())
        t['reference_absolute'] += float((b-y)[mask].abs().sum())
        t['mape_count'] += int(mm.sum())
        t['percentage'] += float(((p-y)[mm].abs()/y[mm]).sum())
        re, rm = region_residual(p, y, geometry, self.config)
        t['region_count'] += int(rm.sum())
        t['region_absolute'] += float(re[rm].abs().sum())

    def result(self):
        t = self.total
        if not t['count']:
            raise ValueError('No valid evaluation targets')
        return dict(mae=t['absolute']/t['count'], rmse=(t['square']/t['count'])**0.5,
            signed_bias=t['signed']/t['count'], negative_fraction=t['negative']/t['count'],
            mape_percent=100*t['percentage']/t['mape_count'] if t['mape_count'] else None,
            valid_count=t['count'], mape_valid_count=t['mape_count'],
            region_mae=t['region_absolute']/t['region_count'] if t['region_count'] else None,
            region_valid_count=t['region_count'], reference_mae=t['reference_absolute']/t['count'])


def metric_contract(config):
    return dict(units='original_flow_units', pooled_accumulation='float64',
        valid_target='finite_and_strictly_greater_than_target_minimum',
        target_minimum=config['target_minimum'], mape_minimum=config['mape_minimum'],
        mape_units='percent', minimum_region_coverage=config['minimum_region_coverage'],
        historical_native_metric_equivalence_claimed=False,
        region_metric='absolute_mean_residual_on_same_observed_node_subset',
        reference_role='last_observed_flow_persistence_not_trained_forecast',
        prediction_clipping=False, negative_predictions_reported=True,
        overlapping_windows_independent=False, low_flow_upper=10.,
        screen_groups='future_truth_for_evaluation_only_not_model_inputs')


def numpy_metrics(prediction, target, config, extra_mask=None):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mask = np.isfinite(y) & (y > config['target_minimum'])
    if extra_mask is not None:
        mask &= np.broadcast_to(extra_mask, y.shape)
    count = int(mask.sum())
    if not count:
        return dict(mae=None, rmse=None, mape_percent=None, signed_bias=None,
                    negative_fraction=None, valid_count=0, mape_valid_count=0)
    if not np.isfinite(p[mask]).all():
        raise ValueError('Nonfinite prediction on valid targets')
    error = p[mask]-y[mask]
    mm = mask & (y >= config['mape_minimum'])
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
        signed_bias=float(error.mean()), negative_fraction=float(np.mean(p[mask] < 0)),
        valid_count=count, mape_valid_count=int(mm.sum()),
        mape_percent=float(100*np.mean(np.abs(p[mm]-y[mm])/y[mm])) if mm.any() else None)


def window_mae(prediction, target, config):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    valid = np.isfinite(y) & (y > config['target_minimum'])
    axes = tuple(range(1, y.ndim))
    total = np.where(valid, np.abs(p-y), 0).sum(axis=axes)
    count = valid.sum(axis=axes)
    return np.divide(total, count, out=np.full(len(y), np.nan), where=count > 0)


def archive_diagnostics(archive, config):
    p, y, b = (np.asarray(archive[k], dtype=np.float64) for k in ('prediction', 'target', 'reference'))
    assignment = np.asarray(archive['assignment'])
    valid = np.isfinite(y) & (y > config['target_minimum'])
    per_region, regional_errors, coherence = {}, [], []
    for r in range(int(assignment.max())+1):
        nodes = assignment == r
        mask = valid[:, :, nodes]
        count = mask.sum(axis=2)
        covered = count >= config['minimum_region_coverage']*int(nodes.sum())
        error = np.where(mask, p[:, :, nodes]-y[:, :, nodes], 0).sum(axis=2)/np.maximum(count, 1)
        regional_errors.extend(np.abs(error[covered]).tolist())
        coherence.append(p[:, :, nodes].mean(axis=2)-archive['regional'][:, :, r])
        per_region[str(r)] = dict(nodes=int(nodes.sum()),
            node_metrics=numpy_metrics(p[:, :, nodes], y[:, :, nodes], config),
            aggregate_mae=float(np.abs(error[covered]).mean()) if covered.any() else None,
            valid_aggregate_cells=int(covered.sum()))
    windows = window_mae(p, y, config)
    support = archive['support'][:, None, :, None].astype(bool)
    long = np.zeros((1, p.shape[1], 1, 1), dtype=bool)
    long[:, 6:12] = True
    groups = dict(
        reported_relation_support=numpy_metrics(p, y, config, support),
        outside_relation_support=numpy_metrics(p, y, config, ~support),
        low_flow_posthoc=numpy_metrics(p, y, config, y <= 10.),
        high_value_ge314=numpy_metrics(p, y, config, y >= 314.),
        long_horizon_7_12=numpy_metrics(p, y, config, long),
        flow_50_200=numpy_metrics(p, y, config, (y >= 50.) & (y < 200.)))
    return dict(overall=numpy_metrics(p, y, config), reference=numpy_metrics(b, y, config),
        per_horizon=[dict(lead=h+1, **numpy_metrics(p[:, h], y[:, h], config)) for h in range(p.shape[1])],
        region_mae=float(np.mean(regional_errors)) if regional_errors else None,
        region_coherence_max_abs=float(np.max(np.abs(np.stack(coherence)))),
        per_region=per_region, groups=groups, window_p95=float(np.nanquantile(windows, .95)),
        window_count=len(windows), independent_event_count=None,
        interpretation='reused_development_windows_no_independent_confirmation_or_causal_impact')
