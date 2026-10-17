"""Bounded engineering evidence: real objective, exact mode, honest OOM."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.chronological import verify_unified_adapter_v03 as verifier
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.unified_incident_adapter_v03 import build_variant
from tests import test_original_isp as fixture


class UnifiedVerificationTests(unittest.TestCase):
    model = fixture.ISPTests.model

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    def batch(self):
        x, report = fixture.ISPTests.batch()
        report['available'] = torch.ones(x.shape[0], dtype=torch.bool)
        report['relation_available'] = torch.ones(x.shape[0], dtype=torch.bool)
        target = torch.rand((2, 12, 4, 1))*.1+1.
        return x, target, report, None

    def scaler(self):
        return self.modules.loader.StandardScaler(8., 2.)

    def test_parser_defaults_exact_requested_stress_batch_and_check_are_distinct(self):
        args = verifier.parse_args(['--output', 'evidence.json'])
        self.assertEqual(args.batch_size, 48)
        self.assertFalse(args.stress)
        stress = verifier.parse_args(['--output', 'evidence.json', '--stress'])
        self.assertTrue(stress.stress)
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            verifier.parse_args(['--output', 'evidence.json', '--stress', '--check'])
        for value in ('0', '-2'):
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                verifier.parse_args(['--output', 'evidence.json', '--batch-size', value])

    def test_zero_recovery_both_modes_rng_and_constructor(self):
        result = verifier.zero_recovery(self.model(), self.batch())
        self.assertEqual(set(result), {'TRAFFIC_CORE', 'H_ONLY', 'R_ONLY', 'BOTH', 'DIRECT'})
        for modes in result.values():
            for row in modes.values():
                self.assertTrue(row['prediction_exact'])
                self.assertTrue(row['dropout_rng_exact'])
                self.assertTrue(row['constructor_rng_exact'])
                self.assertTrue(row['traffic_parameters_and_buffers_exact'])

    def test_activated_counterexamples_are_not_optimizer_training(self):
        result = verifier.activated_interventions(self.model(), self.batch(), scaler=self.scaler(),
            metrics=self.modules.metrics, device='cpu')
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertGreater(result['time_response_abs_max'], 0)
        self.assertGreater(result['graph_response_abs_max'], 0)
        self.assertTrue(result['absent_nan_report_recovers_core_exact'])
        self.assertTrue(result['target_independence_exact'])
        for name in ('report_encoder', 'traffic_summary', 'state', 'graph_head', 'time_head'):
            self.assertGreater(result['perturbed_real_mae_backward']['groups'][name]['l2_norm'], 0)

    def test_real_native_mae_reaches_both_heads_without_parameter_update(self):
        model = build_variant(self.model(), 'BOTH')
        parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        row = verifier.real_mae_backward(model, self.batch(), self.scaler(), self.modules.metrics, 'cpu')
        self.assertEqual(row['objective'], 'native_inverse_transform_masked_mae')
        self.assertTrue(row['true_targets_used_in_objective'])
        self.assertFalse(row['true_targets_passed_to_model'])
        self.assertEqual(row['optimizer_updates'], 0)
        self.assertEqual(row['actual_batch_size'], 2)
        self.assertEqual(row['backward_calls'], 1)
        self.assertGreater(row['groups']['time_head']['l2_norm'], 0)
        self.assertGreater(row['groups']['graph_head']['l2_norm'], 0)
        self.assertTrue(all(row['finite'] for row in row['groups'].values()))
        self.assertIsNone(row['peak_allocated_bytes'])
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, parameters[name]), name)

    def test_real_mae_depends_on_real_targets(self):
        source, batch = self.model(), self.batch()
        a = verifier.real_mae_backward(build_variant(copy.deepcopy(source), 'BOTH'), batch,
            self.scaler(), self.modules.metrics, 'cpu')
        other = (batch[0], batch[1]+5, batch[2], batch[3])
        torch.manual_seed(41)
        b = verifier.real_mae_backward(build_variant(copy.deepcopy(source), 'BOTH'), other,
            self.scaler(), self.modules.metrics, 'cpu')
        self.assertNotEqual(a['loss'], b['loss'])

    def test_all_leads_default_avoids_curriculum_graph_blind_spot(self):
        model = build_variant(self.model(), 'BOTH')
        row = verifier.real_mae_backward(model, self.batch(), self.scaler(), self.modules.metrics, 'cpu',
            curriculum_length=1)
        self.assertEqual(row['curriculum_length'], 1)
        self.assertEqual(row['groups']['graph_head']['l2_norm'], 0)
        self.assertGreater(row['groups']['time_head']['l2_norm'], 0)

    def test_gradient_report_retains_zero_absent_and_nonfinite(self):
        model = build_variant(self.model(), 'BOTH')
        result = verifier.gradient_groups(model)
        self.assertEqual(result['state']['parameters_with_grad'], 0)
        self.assertEqual(result['state']['l2_norm'], 0)
        model.adapter.output.bias.grad = torch.ones_like(model.adapter.output.bias)*float('nan')
        result = verifier.gradient_groups(model)
        self.assertFalse(result['time_head']['finite'])
        self.assertIsNone(result['time_head']['l2_norm'])

    def test_oom_classification_does_not_treat_unrelated_error_as_oom(self):
        self.assertTrue(verifier.is_out_of_memory(torch.cuda.OutOfMemoryError('out of memory')))
        self.assertTrue(verifier.is_out_of_memory(RuntimeError('CUDA out of memory.')))
        self.assertFalse(verifier.is_out_of_memory(ValueError('out of memory field')))
        self.assertFalse(verifier.is_out_of_memory(RuntimeError('shape mismatch')))

    def test_oom_writes_failed_evidence_and_never_retries_with_smaller_batch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary)/'oom.json'
            with patch.object(verifier, 'configure_runtime', return_value={}), \
                    patch.object(verifier, 'verify_original_source', return_value={}), \
                    patch.object(verifier, 'sha256', return_value='hash'), \
                    patch.object(verifier, 'load_data', side_effect=torch.cuda.OutOfMemoryError('out of memory')) as loader, \
                    patch('builtins.print'), self.assertRaises(torch.cuda.OutOfMemoryError):
                verifier.main(['--output', str(output), '--stress', '--batch-size', '48', '--device', 'cpu'])
            result = json.loads(output.read_text())
            self.assertEqual(result['status'], 'UNIFIED_ADAPTER_V03_ENGINEERING_OOM')
            self.assertEqual(result['requested_batch_size'], 48)
            self.assertFalse(result['requested_batch_fallback_permitted'])
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertEqual(loader.call_count, 1)

    def test_existing_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary)/'evidence.json'
            output.write_text('existing')
            with self.assertRaisesRegex(ValueError, 'fresh'):
                verifier.main(['--output', str(output), '--check'])
            self.assertEqual(output.read_text(), 'existing')


if __name__ == '__main__':
    unittest.main()
