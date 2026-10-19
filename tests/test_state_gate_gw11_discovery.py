"""Server-only checks that discovery follows cache provenance rather than scores."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.state_gate_gw11 import discover as subject


def pair(root, probe, tag):
    cache = root/(probe.name+'_gw10_cache_'+tag); cache.mkdir()
    (cache/'cache.json').write_text(json.dumps(dict(schema='gw10_frozen_sum_residual_cache_v1',
        status='GW10_FROZEN_SUM_CACHE_COMPLETE',source_identity_sha256=subject.SOURCE_IDENTITY_SHA256)))
    for name in ('train.npz','val.npz'): (cache/name).write_bytes(b'not loaded during discovery')
    result = root/(probe.name+'_gw10_probe_'+tag); result.mkdir()
    for name in ('model.json','screening.json','protocol.json','report.md'): (result/name).write_text('{}')
    (result/'analysis.json').write_text(json.dumps(dict(schema='gw10_train_only_residual_direction_probe_v1',
        source_certificate=dict(source_identity_sha256=subject.SOURCE_IDENTITY_SHA256),
        cache_metadata_sha256=subject.digest(cache/'cache.json'))))
    return cache,result


class DiscoveryTests(unittest.TestCase):
    def test_completed_result_is_paired_by_cache_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); probe = root/'gw9'; probe.mkdir()
            cache,result = pair(root,probe,'a')
            extra = root/(probe.name+'_gw10_cache_unfinished'); extra.mkdir()
            (extra/'cache.json').write_text('{}')
            self.assertEqual(subject.discover(probe),(cache.resolve(),result.resolve()))

    def test_explicit_mismatching_cache_cannot_silently_choose_another(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); probe = root/'gw9'; probe.mkdir()
            cache,result = pair(root,probe,'a')
            changed = json.loads((cache/'cache.json').read_text()); changed['new_field']='changes hash'
            (cache/'cache.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'No complete matching'):
                subject.discover(probe,cache,result)

    def test_missing_completed_artifact_rejects_partial_result(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); probe = root/'gw9'; probe.mkdir()
            _,result = pair(root,probe,'a'); (result/'model.json').unlink()
            with self.assertRaises(ValueError): subject.discover(probe)

    def test_unknown_source_identity_is_not_selected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); probe = root/'gw9'; probe.mkdir()
            cache,_ = pair(root,probe,'a')
            (cache/'cache.json').write_text(json.dumps(dict(schema='gw10_frozen_sum_residual_cache_v1',
                status='GW10_FROZEN_SUM_CACHE_COMPLETE',source_identity_sha256='another-run')))
            with self.assertRaises(ValueError): subject.discover(probe)


if __name__=='__main__': unittest.main()
