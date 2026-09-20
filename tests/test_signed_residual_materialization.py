"""Tests for frozen-A v7a signed-residual materialization."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from experiments.chronological.materialize_signed_residual import (
    load_protocol, materialize_cohort,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'signed_residual_materialize_v7a.json')
BASE_PROTOCOL = (REPO / 'experiments/chronological' /
                 'incident_branch_materialize_v6a.json')


class ProtocolTests(unittest.TestCase):
    def test_cli_exposes_bounded_engineering_check(self):
        result = subprocess.run(
            [sys.executable,
             'experiments/chronological/materialize_signed_residual.py', '--help'],
            cwd=REPO, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--check', result.stdout)

    def test_protocol_freezes_signed_residual_and_hard_support(self):
        protocol, _ = load_protocol(PROTOCOL, BASE_PROTOCOL)
        self.assertEqual(protocol['inference']['mode'], 'incident_on')
        self.assertEqual(protocol['inference']['signed_residual_definition'],
                         'target_minus_frozen_A_prediction')
        self.assertEqual(protocol['support']['stored_horizons_zero_based_half_open'],
                         [0, 6])
        self.assertTrue(protocol['support']['protected_noncandidate_nodes'])
        self.assertTrue(protocol['information_boundary']['test_split_prohibited'])

    def test_protocol_rejects_future_y_as_an_input(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['information_boundary']['future_Y_may_not_change_inputs_or_cohort'] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'information boundary changed'):
                load_protocol(path, BASE_PROTOCOL)


class _Dataset(torch.utils.data.Dataset):
    scaler = {'mean': 10., 'std': 2.}
    station_ids = np.array([101, 102, 103])

    def __len__(self):
        return 2

    def __getitem__(self, index):
        target = torch.full((12, 3, 1), 15. + index)
        valid = torch.ones((12, 3, 1), dtype=torch.bool)
        return {
            'x': torch.zeros((12, 3, 3)),
            'y_flow': target,
            'y_valid': valid,
            'incident': {
                'report_age_minutes': torch.tensor(5.),
                'forecast_tod': torch.tensor(10),
                'forecast_dow': torch.tensor(1),
                'distances': torch.ones((3, 3)),
            },
            'candidate_mask': torch.tensor([True, False, True]),
            'positive_sample_index': np.int64(10 + index),
        }


class _Model(torch.nn.Module):
    def forward(self, x, incident_data=None):
        return torch.ones((len(x), 12, 3, 1), device=x.device)


class MaterializationTests(unittest.TestCase):
    def test_materialization_stores_target_minus_prediction_and_all_cell_totals(self):
        dataset = _Dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'residuals.npz'
            result = materialize_cohort(
                _Model(), dataset, batch_size=2, device=torch.device('cpu'),
                output_path=path)
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(stored['signed_residual'].shape, (2, 6, 3, 1))
                np.testing.assert_array_equal(stored['signed_residual'][0], 3.)
                np.testing.assert_array_equal(stored['signed_residual'][1], 4.)
                np.testing.assert_array_equal(stored['baseline_prediction'], 12.)
                np.testing.assert_array_equal(
                    stored['baseline_all_absolute_sum'], [108., 144.])
                np.testing.assert_array_equal(stored['baseline_all_valid_count'], 36)
            self.assertAlmostEqual(result['all']['mae_A'], 3.5)


if __name__ == '__main__':
    unittest.main()
