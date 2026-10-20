"""Server-only checks that collection cannot mix scopes or change native axes."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments.state_exchange_multidata.collect import comparison_rows, render


def results_fixture():
    results = {}
    for name in ('Alameda', 'Contra_Costa', 'Orange'):
        rows = {}
        for case, mae in (('ORIGINAL_IGSTGNN', 10.), ('STATE_EXCHANGE', 9.)):
            macro = dict(mae=mae, rmse=mae+2., mape_percent=mae+4.)
            rows[case] = dict(author_metrics=dict(horizon_macro=macro,
                per_horizon=[dict(lead=h, **macro) for h in range(1, 13)]),
                common_mask_pooled=dict(mae=999., rmse=999., mape_percent=999.))
        results[name] = dict(dataset=name, split='test', samples=42, rows=rows,
            parameter_counts=dict(ORIGINAL_IGSTGNN=455220, STATE_EXCHANGE=74000))
    return results


class CollectionTests(unittest.TestCase):
    def test_primary_rows_use_native_macro_not_pooled(self):
        rows = comparison_rows(results_fixture())
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[1]['mae'], 9.)
        self.assertEqual(rows[1]['rmse'], 11.)
        self.assertEqual(rows[1]['mae_gain_percent_vs_original'], 10.)
        self.assertEqual([r['dataset'] for r in rows],
                         ['Alameda', 'Alameda', 'Contra_Costa', 'Contra_Costa', 'Orange', 'Orange'])

    def test_validation_and_test_rows_cannot_be_mixed(self):
        results = results_fixture()
        results['Orange']['split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'mix'):
            comparison_rows(results)

    def test_dataset_names_cannot_be_silently_relabelled(self):
        results = results_fixture()
        results['Alameda']['dataset'] = 'Orange'
        with self.assertRaisesRegex(ValueError, 'mix'):
            comparison_rows(results)

    def test_nonfinite_values_and_zero_denominators_are_rejected(self):
        original = results_fixture()
        for value in (0., float('nan'), float('inf')):
            results = copy.deepcopy(original)
            results['Alameda']['rows']['ORIGINAL_IGSTGNN']['author_metrics']['horizon_macro']['mae'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison_rows(results)

    def test_degradation_is_reported_with_negative_gain(self):
        results = results_fixture()
        results['Orange']['rows']['STATE_EXCHANGE']['author_metrics']['horizon_macro']['mae'] = 11.
        self.assertEqual(comparison_rows(results)[-1]['mae_gain_percent_vs_original'], -10.)

    def test_reports_include_all_three_datasets_all_twelve_horizons_and_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render(root, results_fixture())
            self.assertEqual(len((root/'metrics.csv').read_text().splitlines()), 7)
            self.assertEqual(len((root/'per_horizon.csv').read_text().splitlines()), 73)
            text = (root/'report.md').read_text()
            for term in ('Alameda', 'Contra_Costa', 'Orange', '455220', '74000', 'patience20'):
                self.assertIn(term, text)


if __name__ == '__main__':
    unittest.main()
