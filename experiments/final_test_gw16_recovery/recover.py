"""Resume e5dac19 without editing its frozen code or saved original predictions.

The original model returns a transposed [batch,lead,node,1] prediction. NPZ
preserves values but not its arbitrary strides. Recover that one known layout,
then let the original evaluator enforce exactly the same strict checks.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import uuid


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _value_hash(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def restore_native_layout(prediction):
    import numpy as np
    require(isinstance(prediction, np.ndarray) and prediction.dtype == np.float32
            and prediction.ndim == 4 and prediction.shape[-1] == 1,
            'Expected float32 native [batch,lead,node,1] prediction')
    # Exact layout of frozen original model.py:156-157. This is a copy in
    # [B,N,H,1] order followed by a view; no value is recomputed or rounded.
    restored = prediction.transpose(0, 2, 1, 3).copy(order='C').transpose(0, 2, 1, 3)
    require(restored.shape == prediction.shape and restored.dtype == prediction.dtype
            and _value_hash(restored) == _value_hash(prediction),
            'Layout recovery changed prediction bits')
    return restored


def metric_differences(stored, actual, field=''):
    if isinstance(stored, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(stored) | set(actual)):
            path = field+'.'+key if field else key
            if key not in stored or key not in actual:
                differences.append(dict(field=path, stored=stored.get(key), recomputed=actual.get(key)))
            else:
                differences.extend(metric_differences(stored[key], actual[key], path))
        return differences
    if isinstance(stored, list) and isinstance(actual, list) and len(stored) == len(actual):
        return [item for index, (left, right) in enumerate(zip(stored, actual))
                for item in metric_differences(left, right, field+'['+str(index)+']')]
    if stored == actual:
        return []
    result = dict(field=field, stored=stored, recomputed=actual)
    if isinstance(stored, (int, float)) and isinstance(actual, (int, float)):
        result['absolute_difference'] = abs(stored-actual)
    return [result]


def repair_cached_arrays(arrays, metadata, case, summary):
    """One fixed reconstruction; exact saved-metric equality or fail closed."""
    require(case in ('ORIGINAL_IGSTGNN', 'STATE_EXCHANGE'), 'Unknown frozen test case')
    stored = metadata['author_metrics']
    direct = summary(arrays['prediction'], arrays['target'])
    differences = metric_differences(stored, direct)
    audit = dict(case=case, layout_restored=False,
        prediction_values_bitwise_unchanged=True, strict_metrics_reproduced=not differences,
        serialized_prediction_strides=list(arrays['prediction'].strides),
        direct_metric_differences=differences,
        prediction_values_sha256=_value_hash(arrays['prediction']))
    if not differences:
        return arrays, audit
    require(case == 'ORIGINAL_IGSTGNN',
            'Unexpected STATE cache metric difference: '+json.dumps(differences[:12], ensure_ascii=False))
    restored = dict(arrays, prediction=restore_native_layout(arrays['prediction']))
    replayed = summary(restored['prediction'], restored['target'])
    remaining = metric_differences(stored, replayed)
    require(not remaining,
            'Fixed native layout did not exactly reproduce saved metrics: '
            +json.dumps(remaining[:12], ensure_ascii=False))
    audit.update(layout_restored=True, strict_metrics_reproduced=True,
                 restored_prediction_strides=list(restored['prediction'].strides))
    return restored, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--frozen-code', type=Path, required=True)
    parser.add_argument('--expected-freeze-id', required=True)
    own, evaluation_args = parser.parse_known_args(argv)
    require('--output-dir' in evaluation_args, 'Pass the existing frozen --output-dir explicitly')
    output = Path(evaluation_args[evaluation_args.index('--output-dir')+1]).resolve()
    code = own.frozen_code.resolve()
    # Run old imports from their original absolute paths. Downloading a new
    # evaluator over the old directory would invalidate its frozen source map.
    sys.path.insert(0, str(code))
    entry = importlib.import_module('experiments.final_test_gw16.evaluate')
    state = importlib.import_module('experiments.final_test_gw16.state')
    require(Path(entry.__file__).resolve() == code/'experiments/final_test_gw16/evaluate.py',
            'Recovery imported an evaluator outside the original frozen code directory')
    expected_source = '2ca57f55f5310909b67a7e22a38552ba02fa16ddeb2f0e8606d446f4fd3786a2'
    require(state.sha256(Path(entry.__file__)) == expected_source,
            'Recovery requires the unchanged e5dac19 evaluator, not an edited copy')
    freeze_path = output/'freeze.json'
    record = json.loads(freeze_path.read_text(encoding='utf-8'))
    require(record.get('schema') == state.FREEZE_SCHEMA
            and record['freeze_id'] == own.expected_freeze_id
            and record['freeze_id'] == state.payload_hash(record['payload']),
            'Existing freeze identity differs; never start a replacement test')
    state.verify_files(record['payload']['source_files'])
    original_dir = output/'ORIGINAL_IGSTGNN'
    preserved = state.snapshot_files([freeze_path, original_dir/'metadata.json', original_dir/'predictions.npz'])
    source = state.snapshot_files([Path(__file__)])
    audit_path = output/('serialization_recovery_'+uuid.uuid4().hex+'.json')
    audit = dict(schema='final_test_gw16_serialization_recovery_v1',
        status='RECOVERY_STARTED', original_freeze_id=record['freeze_id'],
        recovery_source=source, preserved_artifacts=preserved,
        original_evaluator_source_changed=False, training_or_selection_changed=False,
        metric_functions_changed=False, metric_tolerance_relaxed=False,
        method='single_fixed_native_prediction_stride_reconstruction_if_needed', cache_reads=[])
    state.write_json(audit_path, audit)
    original_reader = entry.cache_read

    def read_with_layout(directory, freeze_id, case):
        arrays, metadata = original_reader(directory, freeze_id, case)
        for key, expected in (('runtime', record['payload']['runtime']),
                              ('metric_contract', entry.METRIC_CONTRACT)):
            differences = metric_differences(metadata[key], expected, key)
            require(not differences, 'Final-test '+key+' differs: '
                    +json.dumps(differences[:12], ensure_ascii=False))
        # Runtime/metric_contract/data/weights remain under old validate_cached.
        # This callback never writes predictions, metadata or the original freeze.
        arrays, item = repair_cached_arrays(arrays, metadata, case, entry.native_summary)
        common = entry.numpy_metrics(arrays['prediction'], arrays['target'],
                                     entry.METRIC_CONTRACT['common_mask'])
        differences = metric_differences(metadata['common_mask_pooled'], common, 'common_mask_pooled')
        require(not differences, 'Final-test common metrics differ: '
                +json.dumps(differences[:12], ensure_ascii=False))
        audit['cache_reads'].append(dict(directory=str(Path(directory).resolve()), **item))
        state.write_json(audit_path, audit)
        if item['layout_restored']:
            print('ORIGINAL_IGSTGNN cache layout restored; saved metrics reproduced exactly; '
                  'prediction bits unchanged; zero new original forwards', flush=True)
        return arrays, metadata

    entry.cache_read = read_with_layout
    try:
        result = entry.main(evaluation_args)
        state.verify_files(preserved)
        state.verify_files(source)
        audit.update(status='RECOVERY_COMPLETE', original_artifacts_byte_identical=True)
        state.write_json(audit_path, audit)
        return result
    except BaseException as error:
        audit.update(status='RECOVERY_FAILED', error_type=type(error).__name__, error=str(error))
        try:
            state.verify_files(preserved)
            audit['original_artifacts_byte_identical'] = True
        finally:
            state.write_json(audit_path, audit)
        raise
    finally:
        entry.cache_read = original_reader


if __name__ == '__main__':
    main()
