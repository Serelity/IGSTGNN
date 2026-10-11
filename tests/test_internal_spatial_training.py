import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_internal_spatial as training
from experiments.chronological.train import cpu_state, save_checkpoint, state_sha256
from src.models.incident_spatial import configure_spatial_mode
from tests.test_architecture_mechanisms import inputs, tiny_model
from tests.test_incident_spatial import graph


class Dataset:
    def __init__(self):
        self.rows = [dict(sample_index=i, t0='2020-01-01T12:00:00') for i in range(4)]
        self.station_ids = np.arange(4)
        self.scaler = dict(std=2., mean=10.)
        x, incident = inputs()
        self.samples = [dict(x=x[i % 2], incident={k: v[i % 2] for k, v in incident.items()},
                             y_flow=torch.full((12, 4, 1), 9.+i), y_valid=torch.ones(12, 4, 1, dtype=torch.bool))
                        for i in range(4)]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.rows)


def candidate(*args):
    return configure_spatial_mode(tiny_model().requires_grad_(True), 'state_restricted', graph(4, ((0, 1), (1, 2), (2, 3))))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_resume_next_epoch_matches_continuous_including_optimizer_and_scheduler(self):
        protocol = training.read_json(training.HERE/'internal_spatial_is11.json')
        dataset, g = Dataset(), graph(4, ((0, 1), (1, 2), (2, 3)))
        identity = dict(fixture='resume', seed=2025)
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory)/name for name in ('continuous', 'resumed')]
            for root in roots:
                root.mkdir()
            args = SimpleNamespace(worker='B3', run_dir=roots[0], device='cpu', seed=2025, check=True, epochs=2)
            identity['batch_size'] = 2
            with patch.object(training, 'build_arm', side_effect=candidate):
                training.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                args.run_dir, args.epochs = roots[1], 1
                training.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                args.epochs = 2
                training.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
            a, b = [torch.load(root/'B3/last.pt', weights_only=False) for root in roots]
            self.assertEqual(a['global_updates'], b['global_updates'])
            self.assertEqual(a['scheduler_state'], b['scheduler_state'])
            self.assertEqual(a['best_metric'], b['best_metric'])
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key], atol=0, rtol=0)
            for key in a['optimizer_state']['state']:
                for field in a['optimizer_state']['state'][key]:
                    torch.testing.assert_close(a['optimizer_state']['state'][key][field], b['optimizer_state']['state'][key][field], atol=0, rtol=0)
            before = (roots[1]/'B3/last.pt').read_bytes()
            with patch.object(training, 'build_arm', side_effect=candidate):
                training.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
            self.assertEqual(before, (roots[1]/'B3/last.pt').read_bytes())

    def test_bad_checkpoint_identity_fails_before_state_mutation(self):
        model = candidate()
        protocol = training.read_json(training.HERE/'internal_spatial_is11.json')
        optimizer = training.optimizer_for(model, protocol)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1], gamma=.5)
        before = state_sha256(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            save_checkpoint(path, dict(identity={'data': 'changed'}, arm='B3', model_state=cpu_state(model.state_dict())))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                training.restore(path, {'data': 'original'}, 'B3', model, optimizer, scheduler)
        self.assertEqual(before, state_sha256(model.state_dict()))

    def test_five_layers_all_forecast_calls_and_empty_metrics(self):
        model = candidate()
        x, incident = inputs()
        result = model(x, incident_data=incident, return_spatial_details=True)
        self.assertEqual(len(result['spatial']['records']), 20)
        collector = training.Diagnostics()
        collector.update(result['spatial'])
        collector.update(result['spatial'])
        report = collector.result()
        self.assertEqual(len(report['rows']), 40)
        self.assertEqual(report['batches'], 2)
        self.assertEqual(report['b2_stability']['clip_fraction'], 0.)
        self.assertEqual(training.metric_result(training.MetricTotals())['status'], 'unavailable')

    def test_gpu_name_is_audit_field_and_protocol_controls_resume(self):
        protocol = training.read_json(training.HERE/'internal_spatial_is11.json')
        args = SimpleNamespace(protocol=training.HERE/'internal_spatial_is11.json',
                               graph_dir=training.HERE/'internal_spatial_graph_is11', seed=2025,
                               check=False, threads=3, device='cpu')
        with patch.object(training, 'source_fingerprints', return_value={'source': 'hash'}):
            identity = training.make_identity(args, protocol, {'data': 'hash'}, {'graph': 'candidate'})
        self.assertNotIn('gpu', identity['runtime'])
        self.assertEqual(identity['batch_size'], 48)
        self.assertFalse(identity['test_read'])


if __name__ == '__main__':
    unittest.main()
