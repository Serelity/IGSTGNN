"""Reproduce CPU integration, full forecast gradients and real-data checkpoint checks.

Runs only development train/validation rows. It does not benchmark predictive gain
or certify online report availability. Outputs a compact verification artifact.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch

from experiments.chronological import train_b3_r as training
from experiments.chronological.smoke import sha256
from src.utils.chronological import masked_flow_mae
from tests.test_b3_r import controlled_forecast


def main():
    tempfile.tempdir = str(REPO)
    output = Path(__file__).with_name('igstgnn_b3_r_implementation_20261012.json')
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ['tests.test_b3_r', 'tests.test_incident_spatial', 'tests.test_internal_spatial_training'])
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    output.with_suffix('.tests.txt').write_text(log.getvalue(), encoding='utf-8')
    if not tests.wasSuccessful():
        raise ValueError('Unit/regression suite failed; see verification test log')
    print(f'Unit/regression checks: {tests.testsRun} passed', flush=True)
    gradients = []
    with controlled_forecast() as (model, x, report):
        optimizer = torch.optim.SGD(model.report_spatial_state.D.parameters(), lr=.1, weight_decay=0.)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(x, incident_data=report, return_spatial_details=True)
            prediction = result['prediction']
            loss = masked_flow_mae(prediction, prediction.detach()+1., torch.ones_like(prediction, dtype=torch.bool))
            loss.backward()
            d = model.report_spatial_state.D
            gradients.append(dict(step=step+1, D_hidden_task_gradient_l1=float(d[0].weight.grad.abs().sum()+d[0].bias.grad.abs().sum()),
                                  D_last_task_gradient_l1=float(d[-1].weight.grad.abs().sum()+d[-1].bias.grad.abs().sum()),
                                  capacity_hits=sum(float(r[:, 11].sum()) for r in result['spatial']['records']),
                                  sending_receiving_hits=sum(float(r[:, 12:14].sum()) for r in result['spatial']['records'])))
            optimizer.step()
        with torch.inference_mode():
            on = model(x, incident_data=report)
            off = model(x, incident_data=report, spatial_report_enabled=False)
        response = float((on-off).abs().max())
    print('Full forecast MAE to D: controlled two-step gradients passed', flush=True)

    args = training.parse_args(['--check', '--device', 'cpu', '--arms', 'B3', 'B3R', 'B3R_OFF'])
    command_args = ['--check', '--device', 'cpu', '--run-dir', str(args.run_dir),
                    '--data-dir', str(args.data_dir), '--arms', *args.arms]
    training.main(command_args)
    before = {arm: sha256(args.run_dir/arm/'last.pt') for arm in args.arms}
    print('Repeating completed request; checkpoints must remain byte-identical', flush=True)
    training.main(command_args)
    after = {arm: sha256(args.run_dir/arm/'last.pt') for arm in args.arms}
    if before != after:
        raise ValueError('Completed request modified a checkpoint')
    init = training.legacy.read_json(args.run_dir/'initialization.json')
    comparisons = training.legacy.read_json(args.run_dir/'latest_comparison.json')
    arms = {}
    for arm in args.arms:
        summary = training.legacy.read_json(args.run_dir/arm/'summary.json')
        history = training.legacy.read_json(args.run_dir/arm/'history.json')
        phase = history[0]['validation']['mechanisms']
        arms[arm] = dict(completed_epoch=summary['completed_epoch'], updates=summary['global_updates'],
                         parameters=summary['parameters']['total_parameters'], diagnostic_rows=len(phase['rows']),
                         validation_batches=phase['batches'], first_batch_gradient_l1=history[0]['train']['first_batch_gradient_l1'])
        if arm != 'B3':
            arms[arm].update(capacity_evaluations=phase['capacity_evaluations'], raw_delta_status=phase['raw_delta_status'],
                             node_capacity_positions=phase['capacity_statistics']['node_capacity:all_nodes']['count'])
    frozen = {
        'src/models/igstgnn.py': '07b1eb80b0b7f36ebae501a2d9f52c019ef659c3bb5dbf138f8978c47fb1831e',
        'src/models/incident_spatial.py': '42216d06cbfe6356d886cb6983efa03c189c7d0133b6eace4cbbe389fae81ab1',
        'experiments/chronological/train_internal_spatial.py': '371f03212141d6cb2d8599fcc8daa0804c4f87fc75e283fd885312b69b6663d4',
        'experiments/chronological/internal_spatial_is11.json': 'ddc254413a24285609e94767bb04962bb79e340fe3da2ad74114f786bf69284c',
    }
    observed = {name: sha256(REPO/name) for name in frozen}
    if observed != frozen:
        raise ValueError('Original B0-B3 source/config changed')
    payload = dict(status='B3R_IMPLEMENTATION_ENGINEERING_PASS', unit_regression_tests=tests.testsRun,
                   test_failures=0, test_errors=0, environment=training.legacy.environment(torch.device('cpu')),
                   source_sha256=training.source_fingerprints(), old_source_unchanged_sha256=observed,
                   run_dir=str(args.run_dir), real_data_scope='first_four_train_and_validation_rows_only', arms=arms,
                   common_initial_state_sha256=init['mapping']['retained_state_sha256'],
                   canonical_spatial_off_capacity_abs_max=init['canonical_spatial_off_capacity_abs_max'],
                   canonical_spatial_off_prediction_abs_max=init['canonical_spatial_off_prediction_abs_max'],
                   initial_on_off_prediction_abs_max=init['initial_on_off_prediction_abs_max'],
                   controlled_full_forecast_gradient_steps=gradients, controlled_prediction_on_off_abs_max=response,
                   gradient_fixture='fixed P endpoint logits; actual D, two exchange steps, throughput readout, full IGSTGNN forecast and masked MAE',
                   checkpoint_sha256_before_repeat=before, checkpoint_sha256_after_repeat=after,
                   repeated_completed_request_preserves_checkpoints=True,
                   continuation_equivalence='two synthetic epochs continuous vs checkpoint resume: exact model/optimizer/scheduler equality',
                   comparison_scope=comparisons['scope'], server_full_epoch_completed=False,
                   online_time_audit_pass=False, predictive_gain_claim=False, test_read=False)
    training.legacy.write_json(output, payload)
    print('Verification artifact: '+str(output), flush=True)


if __name__ == '__main__':
    main()
