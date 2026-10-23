"""The paper row assembler refuses mixed splits, versions or incomplete BESTs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.pase_compact_main import summarize
from experiments.position_state_gw16.common import digest_json, write_json


class SummaryTests(unittest.TestCase):
    def test_cache_scope_and_budget_checked_before_reading_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root/'Orange'/'evaluation_test_s2025'
            directory.mkdir(parents=True)
            ev = summarize.main_evaluate
            role = dict(completed_training_epochs=100, full_protocol_reached=True,
                        selection='best', selected_on='validation_pooled_mae')
            original = dict(schema=ev.SCHEMA, dataset='Orange', split='test',
                engineering_only=False, arms=['FULL'], seeds=[2025],
                metric_contract=ev.METRIC_CONTRACT, evaluation_source={'source':'fixed'},
                records={'s2025__FULL':{'role':role}})
            def save(value):
                value = copy.deepcopy(value)
                value['frozen_sha256'] = digest_json(value)
                write_json(directory/'frozen.json', value)
            save(original)
            with patch.object(ev, 'evaluation_sources', return_value={'source':'fixed'}), \
                 patch.object(ev, 'read_cache', return_value={'test_evaluated':True}) as cache:
                summarize.read_result(root, 'Orange')
                self.assertEqual(cache.call_count, 1)
                for change in ({'split':'val'}, {'dataset':'Alameda'}, {'arms':['UP_ONLY']},
                               {'schema':summarize.cc_evaluate.SCHEMA}, {'engineering_only':True}):
                    save(dict(original, **change))
                    with self.assertRaises(ValueError):
                        summarize.read_result(root, 'Orange')
                bad = copy.deepcopy(original)
                bad['records']['s2025__FULL']['role']['completed_training_epochs'] = 99
                save(bad)
                with self.assertRaises(ValueError):
                    summarize.read_result(root, 'Orange')
                self.assertEqual(cache.call_count, 1)


if __name__ == '__main__':
    unittest.main()
