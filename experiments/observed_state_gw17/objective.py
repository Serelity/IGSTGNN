"""Matched forecast + pre-correction historical prediction losses in raw units."""
import numpy as np
import torch

from experiments.position_state_gw16.objective import (
    PooledStatistics, numpy_metrics, valid_target, window_mae, region_residual,
    loss_terms as forecast_loss_terms, archive_diagnostics as forecast_diagnostics,
    metric_contract as forecast_contract)
from .common import require


def historical_loss(output, geometry, config):
    prediction, target = output['history_prediction'], output['history_target'].detach()
    valid = output['history_valid'].bool() & valid_target(target, config)
    require(prediction.shape == target.shape == valid.shape,
            'Historical prediction/observed target/validity axes differ')
    require(bool(torch.isfinite(prediction).all()), 'Nonfinite historical pre-correction prediction')
    # Empty adjacent-valid history is reported as no support, never a fabricated
    # success. Its differentiable zero contributes no auxiliary optimization.
    zero = prediction.sum()*0
    node = (prediction-target).abs()[valid].mean() if bool(valid.any()) else zero
    coverage = geometry.pool(valid.to(prediction.dtype))
    residual = torch.where(valid, prediction-target, torch.zeros_like(prediction))
    error = geometry.pool(residual)/coverage.clamp_min(1e-12)
    covered = coverage >= config['minimum_region_coverage']
    region = error.abs()[covered].mean() if bool(covered.any()) else zero
    return node, region, node+config['region_loss_weight']*region


def loss_terms(output, target, geometry, config, stage='joint'):
    future = forecast_loss_terms(output, target, geometry, config, stage=stage)
    node, region, history = historical_loss(output, geometry, config)
    return dict(future, forecast_loss=future['loss'], history_node=node,
        history_region=region, history_loss=history,
        loss=future['loss']+config['history_loss_weight']*history)


def metric_contract(config):
    return dict(forecast_contract(config),
        schema='gw17_forecast_and_preupdate_history_metrics_v1',
        history_scope='11_observed_X_transitions_prediction_before_current_flow_correction',
        history_validity=config['observation_validity'], history_loss_weight=config['history_loss_weight'],
        history_uses_future_Y=False, history_report_fields_used=False,
        history_post_correction_reconstruction_is_prediction=False,
        both_new_arms_share_all_loss_weights=True, test_evaluated=False)


def archive_diagnostics(archive, config):
    result = forecast_diagnostics(archive, config)
    p, y, valid = (archive[key] for key in ('history_prediction', 'history_target', 'history_valid'))
    require(p.shape == y.shape == valid.shape and p.shape[1] == config['history']-1,
            'Historical archive axes differ from the observed 11 transitions')
    require(np.isin(valid, [0, 1]).all(), 'Historical validity archive is not binary')
    result.update(history_preupdate=numpy_metrics(p, y, config, valid.astype(bool)),
        historical_transitions_per_forward=config['history']-1,
        forecast_transitions_per_forward=config['horizon'],
        core_transitions_per_forward=config['history']-1+config['horizon'],
        interpretation='single_seed_development_of_frozen_test_successor_no_new_test')
    return result
