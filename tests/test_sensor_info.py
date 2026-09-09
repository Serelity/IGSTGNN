import csv
import io
import logging
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from src.utils.dataloader import _load_sensor_info


class SensorInfoTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.log = io.StringIO()
        self.logger = logging.Logger(self.id())
        self.logger.addHandler(logging.StreamHandler(self.log))

    def load_rows(self, rows):
        fields = [
            'Type', 'Sensor Type', 'Road Width', 'Lane Width',
            'Design Speed Limit', 'Surface', 'Roadway Use',
        ]
        with (Path(self.temp_dir.name) / 'sensors.csv').open(
                'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        args = types.SimpleNamespace(use_sensor_info=True, dataset='test')
        data = _load_sensor_info(self.temp_dir.name, len(rows), args, self.logger)
        return data, args

    def test_lane_width_and_design_speed_keep_unit_bearing_node_differences(self):
        data, _ = self.load_rows([
            {'Lane Width': '12.0 ft', 'Road Width': '60 ft', 'Design Speed Limit': '70 mph'},
            {'Lane Width': '11.0 ft', 'Road Width': '36 ft', 'Design Speed Limit': '65 mph'},
        ])

        np.testing.assert_allclose(data['road_width'], [3.6576, 3.3528], rtol=1e-6)
        np.testing.assert_allclose(data['speed_limit'], [112.65408, 104.60736], rtol=1e-6)

    def test_bare_numbers_and_explicit_canonical_units_are_not_rescaled(self):
        data, _ = self.load_rows([
            {'Lane Width': '3.5 m', 'Design Speed Limit': '90 km/h'},
            {'Lane Width': '3.2', 'Design Speed Limit': '50'},
        ])

        np.testing.assert_allclose(data['road_width'], [3.5, 3.2], rtol=1e-6)
        np.testing.assert_allclose(data['speed_limit'], [90.0, 50.0], rtol=1e-6)

    def test_categories_use_roadway_type_instead_of_detector_technology(self):
        data, args = self.load_rows([
            {'Type': 'Mainline', 'Sensor Type': 'loops', 'Surface': 'Bridge Deck', 'Roadway Use': 'No Special Features'},
            {'Type': 'Mainline', 'Sensor Type': 'magnetometers', 'Surface': 'Bridge Deck', 'Roadway Use': 'HOV'},
            {'Type': 'Offramp', 'Sensor Type': 'radar', 'Surface': 'Asphalt', 'Roadway Use': 'No Special Features'},
        ])

        np.testing.assert_array_equal(data['sensor_type'], [0, 0, 1])
        np.testing.assert_array_equal(data['surface'], [0, 0, 1])
        np.testing.assert_array_equal(data['roadway_use'], [0, 1, 0])
        self.assertEqual(args.sensor_type_size, 2)
        self.assertEqual(args.surface_size, 2)
        self.assertEqual(args.roadway_use_size, 2)

    def test_missing_numbers_use_reported_defaults_even_when_only_units_remain(self):
        data, _ = self.load_rows([
            {'Lane Width': '', 'Design Speed Limit': ''},
            {'Lane Width': 'ft', 'Design Speed Limit': 'mph'},
        ])

        np.testing.assert_array_equal(data['road_width'], [0.0, 0.0])
        np.testing.assert_array_equal(data['speed_limit'], [50.0, 50.0])
        output = self.log.getvalue()
        self.assertRegex(output, r'Lane Width.*m.*missing=2.*default=0')
        self.assertRegex(output, r'Design Speed Limit.*km/h.*missing=2.*default=50')

    def test_invalid_nonempty_values_fail_with_the_source_field(self):
        for field, value in [
            ('Lane Width', '12 yards'),
            ('Lane Width', 'unknown'),
            ('Lane Width', 'nan'),
            ('Design Speed Limit', '70 knots'),
            ('Design Speed Limit', 'inf'),
        ]:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    self.load_rows([{field: value}])


if __name__ == '__main__':
    unittest.main()
