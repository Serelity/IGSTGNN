"""Fixed descriptive gate; never selects another arm or starts model training."""
import numpy as np

CANDIDATE = 'JOINT'
REQUIRED_ARMS = ('SUM', 'GLOBAL', 'HORIZON', 'NODE_HORIZON', 'ORIGIN',
                 'OWN2', 'NEIGHBOR2', 'OWN4', 'JOINT')
MIN_RELATIVE_GAIN = .005
AUDIT_FIT_STOP = 1826
AUDIT_SCORE_START = 1954
TRAIN_COUNT = 3909


def metrics(prediction, y, valid):
    prediction, y, valid = map(np.asarray, (prediction, y, valid))
    if (prediction.shape != y.shape or valid.shape != y.shape or valid.dtype != np.bool_
            or not valid.any() or not np.isfinite(prediction).all() or not np.isfinite(y).all()):
        raise ValueError('Finite matching predictions and a nonempty Boolean mask are required')
    error = prediction[valid].astype(np.float64)-y[valid].astype(np.float64)
    absolute = np.abs(error); target = y[valid].astype(np.float64)
    nonzero = target != 0.
    return dict(valid_count=int(valid.sum()), mae=float(absolute.mean()),
        rmse=float(np.sqrt(np.mean(error*error))), signed_bias=float(error.mean()),
        mape_fraction=float(np.mean(absolute[nonzero]/np.abs(target[nonzero]))) if nonzero.any() else None)


def paired(before, after, y, valid):
    error_before = np.abs(np.asarray(before)[valid].astype(np.float64)-np.asarray(y)[valid])
    error_after = np.abs(np.asarray(after)[valid].astype(np.float64)-np.asarray(y)[valid])
    if not error_before.size: return dict(valid_count=0, mae_gain=None, relative_gain=None)
    gain = error_before-error_after
    help_value, harm_value = float(np.maximum(gain,0).mean()), float(np.maximum(-gain,0).mean())
    net = float(gain.mean()); movement = help_value+harm_value
    return dict(valid_count=int(gain.size),mae_gain=net,
        relative_gain=net/float(error_before.mean()) if error_before.mean()>0 else None,
        gross_help=help_value,gross_harm=harm_value,
        cancellation=1.-abs(net)/movement if movement else None)


def screen(validation, audit, groups):
    if set(validation) != set(REQUIRED_ARMS) or set(audit) != set(REQUIRED_ARMS):
        raise ValueError('Exactly the predefined arms must be reported')
    candidate = validation[CANDIDATE]['mae']
    checks = {}
    for control in ('SUM','ORIGIN'):
        baseline = validation[control]['mae']
        checks['validation_relative_gain_vs_'+control+'_at_least_0_005'] = (
            baseline>0 and (baseline-candidate)/baseline >= MIN_RELATIVE_GAIN)
    for control in ('OWN2','NEIGHBOR2','OWN4'):
        checks['validation_beats_'+control] = candidate < validation[control]['mae']
    for control in ('ORIGIN','OWN4'):
        checks['training_order_audit_beats_'+control] = audit[CANDIDATE]['mae'] < audit[control]['mae']
    for name in ('high_value_ge314','long_horizon_7_12','flow_50_200'):
        row = groups[name]
        checks[name+'_no_worse_than_SUM'] = (row[CANDIDATE]['valid_count']>0
            and row[CANDIDATE]['mae'] <= row['SUM']['mae'])
    checks = {name:bool(value) for name,value in checks.items()}
    return dict(status='GW10_RESIDUAL_PROBE_SCREEN_COMPLETE',candidate=CANDIDATE,checks=checks,
        eligible_for_new_model_design=all(checks.values()), candidate_for_confirmation=False,
        interpretation='train_fitted_lookup_development_screen_not_independent_confirmation',
        training_order_audit_scope='new_lookup_mapping_only_backbone_previously_trained_on_these_labels',
        backbone_out_of_sample_audit=False,
        winner_selection=False,validation_hyperparameter_search=False,
        GW9_joint_training_authorized=False, backbone_updates=0,SUM_head_updates=0,
        screening_failure_does_not_authorize_retuning=True,test_evaluated=False)
