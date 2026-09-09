#!/usr/bin/env python3
"""Audit exact frame overlap in the actual IGSTGNN split files, without training.

Only load trusted publication/user-provided .npy files: object arrays use pickle.
This checks matching observations, not absolute dates or incident identity.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_splits import load_samples


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame):
    array = np.ascontiguousarray(frame)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode('ascii'))
    digest.update(array.tobytes())
    return digest.digest()


def check_overlap(dataset_dir):
    paths = {split: dataset_dir / f'incident_{split}.npy'
             for split in ('train', 'val', 'test')}
    paths['stats'] = dataset_dir / 'incident_stats.npz'
    fingerprints = {key: {'path': str(path.resolve()), 'sha256': file_digest(path)}
                    for key, path in paths.items()}
    counts, frames, y_counts = {}, {}, {}
    for split in ('train', 'val', 'test'):
        samples = load_samples(paths[split])
        counts[split] = len(samples)
        frames[split] = {'x': set(), 'y': set()}
        y_counts[split] = Counter()
        for sample in samples:
            for part in ('x', 'y'):
                for frame in sample[part + '_data']:
                    key = frame_digest(frame)
                    frames[split][part].add(key)
                    if part == 'y':
                        y_counts[split][key] += 1

    pairs = {}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        left_xy = frames[left]['x'] | frames[left]['y']
        right_xy = frames[right]['x'] | frames[right]['y']
        pair = {
            'overlapping_unique_xy_frames': len(left_xy & right_xy),
            'overlapping_unique_frames_by_part': {
                a + '_vs_' + b: len(frames[left][a] & frames[right][b])
                for a in ('x', 'y') for b in ('x', 'y')
            },
            'right_y_frame_occurrences': sum(y_counts[right].values()),
            'right_y_matches_left_xy_occurrences': sum(
                count for key, count in y_counts[right].items() if key in left_xy),
        }
        for part in ('x', 'y'):
            pair[f'right_y_matches_left_{part}_occurrences'] = sum(
                count for key, count in y_counts[right].items()
                if key in frames[left][part])
        pairs[left + '_vs_' + right] = pair

    return {
        'dataset': dataset_dir.name,
        'fingerprints': fingerprints,
        'samples': counts,
        'frame_definition': 'SHA-256 of shape, dtype and full node/channel frame bytes',
        'has_cross_split_frame_overlap': any(
            pair['overlapping_unique_xy_frames'] > 0 for pair in pairs.values()),
        'pairs': pairs,
        'limitations': [
            'Uses actual split files; does not infer their ordering from incident_all.npy.',
            'Identical frames do not establish original calendar dates or incident identity.',
            'No exact overlap does not certify chronological or event-independent evaluation.',
            'Different normalization or dtype can hide shared underlying observations.',
            'Source data, normalization statistics and split assignments are not modified.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path,
                        help='Create a new JSON report; existing files are never overwritten.')
    parser.add_argument('--fail-on-overlap', action='store_true',
                        help='Exit with status 2 on any cross-split exact frame overlap.')
    args = parser.parse_args()
    report = check_overlap(args.data_root / args.dataset)
    serialized = json.dumps(report, indent=2) + '\n'
    if args.output is not None:
        try:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(serialized)
        except FileExistsError:
            parser.error(f'Output already exists; refusing to overwrite: {args.output}')
    print(serialized, end='')
    return 2 if args.fail_on_overlap and report['has_cross_split_frame_overlap'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
