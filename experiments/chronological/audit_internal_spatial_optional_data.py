"""Independent IS1.1 optional-data audit: existing training X only, no model or target reads."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(directory):
    meta = read(directory/'summary.json')
    path = directory/'train_history.npy'
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.ndim != 4 or array.shape[1] != 12 or array.shape[-1] != 3:
        raise ValueError('Expected training X [samples,12,nodes,3] only')
    expected = meta.get('files', {}).get('train_history.npy') or meta.get('outputs_sha256', {}).get('train_history.npy')
    expected = expected or meta.get('outputs', {}).get('train_history.npy', {}).get('sha256')
    actual = sha(path)
    if expected is not None and expected != actual:
        raise ValueError('Training history fingerprint mismatch')
    counts = np.zeros(3, dtype=np.int64)
    zeros = counts.copy()
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    total = int(np.prod(array.shape[:-1]))
    for start in range(0, array.shape[0], 32):
        block = np.asarray(array[start:start+32])
        valid = np.isfinite(block) & (block >= 0)
        counts += valid.sum((0, 1, 2))
        zeros += (valid & (block == 0)).sum((0, 1, 2))
        minimum = np.minimum(minimum, np.where(valid, block, np.inf).min((0, 1, 2)))
        maximum = np.maximum(maximum, np.where(valid, block, -np.inf).max((0, 1, 2)))
    return dict(status='IS11_OPTIONAL_TRAIN_X_AUDIT_COMPLETE', summary_sha256=sha(directory/'summary.json'),
                train_history_sha256=actual, source_hash_verified=expected is not None, shape=list(array.shape),
                channel_semantics=meta.get('channel_semantics'), source_cache=meta.get('source_cache'),
                channels=[dict(index=i, cells=total, valid_nonnegative=int(counts[i]), zero=int(zeros[i]),
                               missing_fraction=1-int(counts[i])/total,
                               minimum=float(minimum[i]) if counts[i] else None,
                               maximum=float(maximum[i]) if counts[i] else None) for i in range(3)],
                history_slots='0:12 materialized; excluded gap and Y absent from this array',
                timing_evidence='existing materialization metadata only; no new source-time certification',
                source_imputation_certified=False, independent_observation_entities_certified=False,
                online_report_versions_certified=False, physical_units_certified=False,
                first_phase_inputs_changed=False, future_Y_read=False, validation_test_read=False,
                interpretation='Finite arrays do not prove independent sensors or absence of upstream imputation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new optional-data audit output')
    report = audit(args.history_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
