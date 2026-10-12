"""Retained-weight certificates permit fair comparisons without faking hashes."""
from pathlib import Path
import tempfile
import unittest

from experiments.chronological import analyze_model_evaluation as evaluation
from tests.test_model_evaluation import synthetic_run, write_json, canonical_sha256, REPO


def minimal_fixture(root):
    synthetic_run(root, arms=['B3R_MIN'])
    sources = evaluation.Sources()
    identity = sources.read_json(root/'identity.json')
    certificate = sources.read_json(root/'initialization.json')
    identity['schema'] = evaluation.MINIMAL_SCHEMA
    certificate.update(status=evaluation.MINIMAL_CERTIFICATE,
                       retained_state_exact_verified=True, reference_seed=identity['seed'],
                       reference_common_initial_sha256='a'*64,
                       reference_source_sha256=dict(identity['source_sha256']))
    certificate['arms']['B3R_MIN']['common_initial_sha256'] = 'f'*64
    identity['initialization_sha256'] = canonical_sha256(certificate)
    write_json(root/'identity.json', identity)
    write_json(root/'initialization.json', certificate)
    return identity, certificate


class MinimalEvaluationTests(unittest.TestCase):
    def test_pruned_hash_stays_distinct_and_retained_mapping_allows_old_reference(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            minimal_fixture(root/'minimal')
            synthetic_run(root/'reference', arms=['B3R'])
            sources = evaluation.Sources()
            minimal = evaluation.load_run(root/'minimal', sources, False)
            reference = evaluation.load_run(root/'reference', sources, False)
            self.assertNotEqual(minimal['initialization_evidence']['common_initial_sha256'],
                                reference['initialization_evidence']['common_initial_sha256'])
            result = evaluation.compatibility(minimal, reference)
            self.assertTrue(result['paired_allowed'])
            self.assertEqual(result['initialization_comparison'],
                             'fresh_reference_fingerprint_and_exact_retained_state_mapping')
            self.assertFalse(minimal['initialization_evidence']['full_parameter_equality_claim'])
            self.assertTrue(evaluation.compatibility(reference, minimal)['paired_allowed'])
            summary = evaluation.analyze(root/'minimal', root/'analysis', 'B3R_MIN', 2,
                                         root/'reference', ['B3R'], False)
            self.assertEqual(summary['new_inference_calls'], 0)
            self.assertEqual(len(summary['comparisons']), 2)

    def test_bridge_mismatch_rejects_pair_instead_of_relabeling_initialization(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            identity, certificate = minimal_fixture(root/'minimal')
            synthetic_run(root/'reference', arms=['B3R'])
            certificate['reference_common_initial_sha256'] = 'b'*64
            identity['initialization_sha256'] = canonical_sha256(certificate)
            write_json(root/'minimal'/'initialization.json', certificate)
            write_json(root/'minimal'/'identity.json', identity)
            sources = evaluation.Sources()
            result = evaluation.compatibility(evaluation.load_run(root/'minimal', sources, False),
                                              evaluation.load_run(root/'reference', sources, False))
            self.assertFalse(result['paired_allowed'])
            self.assertIn('common_initial_sha256', [r['field'] for r in result['differences']])

    def test_missing_or_false_mapping_and_unbound_certificate_are_rejected(self):
        for fault in ('mapping', 'seed', 'source', 'source_absent', 'digest', 'bridge'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(dir=REPO) as directory:
                root = Path(directory)/'minimal'
                identity, certificate = minimal_fixture(root)
                if fault == 'mapping':
                    certificate['retained_state_exact_verified'] = False
                elif fault == 'seed':
                    certificate['reference_seed'] += 1
                elif fault == 'source':
                    certificate['reference_source_sha256']['src/models/igstgnn.py'] = '0'*64
                elif fault == 'source_absent':
                    certificate['reference_source_sha256'].pop('src/models/igstgnn.py')
                elif fault == 'digest':
                    identity.pop('initialization_sha256')
                else:
                    certificate.pop('reference_common_initial_sha256')
                if fault != 'digest':
                    identity['initialization_sha256'] = canonical_sha256(certificate)
                write_json(root/'identity.json', identity)
                write_json(root/'initialization.json', certificate)
                with self.assertRaisesRegex(ValueError, 'Minimal'):
                    evaluation.load_run(root, evaluation.Sources(), False)


if __name__ == '__main__':
    unittest.main()
