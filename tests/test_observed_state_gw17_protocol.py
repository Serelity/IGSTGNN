"""Server-only guards for fixed budget, immutable reference and no test loader."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.observed_state_gw17 import common, data, reference
from experiments.chronological.original_igstgnn_runtime import RELEASE


class GW17ProtocolTests(unittest.TestCase):
    def test_configuration_json_exactly_matches_locked_protocol(self):
        self.assertEqual(common.load_config(), common.LOCKED_CONFIG)
        for key, value in [('max_epochs', 105), ('history_loss_weight', .5),
                           ('early_stopping', True), ('init_readout', 'random')]:
            config = copy.deepcopy(common.LOCKED_CONFIG)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.validate_config(config)

    def test_engineering_is_explicitly_separate_and_cannot_raise_the_budget(self):
        config = copy.deepcopy(common.LOCKED_CONFIG)
        config.update(max_epochs=3, lr_milestones=[])
        common.validate_config(config, engineering=True)
        with self.assertRaises(ValueError):
            common.validate_config(config)
        with self.assertRaises(ValueError):
            common.phase_for(4, config)

    def test_development_loader_cannot_open_test(self):
        loader = object()
        material = SimpleNamespace(loaders=dict(val_loader=loader))
        self.assertEqual(data.validation_loader(material, 'val', 48, None), (loader, 3909))
        with self.assertRaisesRegex(ValueError, 'no test'):
            data.validation_loader(material, 'test', 48, None)

    def test_new_output_cannot_intersect_old_training_identity_or_data(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root/'old'
            old.mkdir()
            (old/'identity.json').write_text(json.dumps(dict(schema='position_state_gw16_training_v1')))
            with self.assertRaises(ValueError):
                common.protected_output(old/'new', root/'data')
            with self.assertRaises(ValueError):
                common.protected_output(root/'data'/'new', root/'data')
            self.assertEqual(common.protected_output(root/'new', root/'data'), (root/'new').resolve())

    def fixture(self, directory):
        for name in ('metadata.json', 'predictions.npz'):
            (directory/name).write_bytes(b'authenticated synthetic '+name.encode())
        arrays = dict(prediction=np.zeros((837, 1), dtype=np.float32),
                      station_ids=np.arange(496, dtype=np.int64),
                      assignment=np.zeros(496, dtype=np.int64))
        metadata = dict(arm='STATE_EXCHANGE', engineering_only=False, split='val', test_evaluated=False,
            checkpoint_role=dict(selection='best', full_protocol_reached=True, completed_training_epochs=100),
            validation_replay=dict(passed=True), identity=dict(data=dict(hashes=RELEASE.copy(), position={})),
            identity_sha256='certified-gw16', diagnostics=dict(overall=dict(mae=12.528962405)))
        return arrays, metadata

    def test_reference_accepts_only_full_validation_best_and_detects_later_mutation(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays, metadata = self.fixture(directory)
            with patch.object(reference, 'read_archive', return_value=(arrays, metadata)):
                _, _, record = reference.read_reference(directory)
            self.assertEqual(record['metrics']['mae'], 12.528962405)
            self.assertEqual(record['historical_model_forward_calls'], 0)
            (directory/'predictions.npz').write_bytes(b'modified after certification')
            with self.assertRaisesRegex(ValueError, 'reference changed'):
                reference.verify_reference(record)

    def test_reference_rejects_last_partial_other_arm_and_test_archives(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays, initial = self.fixture(directory)
            for mutate in (
                lambda m:m.update(arm='NATIVE_POS'),
                lambda m:m.update(engineering_only=True),
                lambda m:m.update(split='test'),
                lambda m:m.update(test_evaluated=True),
                lambda m:m['checkpoint_role'].update(selection='last'),
                lambda m:m['checkpoint_role'].update(completed_training_epochs=99),
            ):
                metadata = copy.deepcopy(initial)
                mutate(metadata)
                with patch.object(reference, 'read_archive', return_value=(arrays, metadata)), self.assertRaises(ValueError):
                    reference.read_reference(directory)

    def test_reference_requires_matching_position_station_and_region_axes(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays, metadata = self.fixture(directory)
            material = SimpleNamespace(identity=dict(hashes=RELEASE.copy()), station_ids=np.arange(496),
                assignment=np.zeros(496, dtype=np.int64), position_identity={})
            with patch.object(reference, 'read_archive', return_value=(arrays, metadata)):
                reference.read_reference(directory, material)
                material.assignment[0] = 1
                with self.assertRaisesRegex(ValueError, 'identities differ'):
                    reference.read_reference(directory, material)


if __name__ == '__main__':
    unittest.main()
