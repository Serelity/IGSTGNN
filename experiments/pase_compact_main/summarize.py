"""Assemble three authenticated FULL test caches; no model forwards or fitting."""
import argparse
import csv
from pathlib import Path

from experiments.pase_compact_v3 import evaluate as cc_evaluate
from . import evaluate as main_evaluate
from .state import digest_json, read_json, require, write_json


def read_result(root, dataset):
    directory = Path(root)/dataset/'evaluation_test_s2025'
    evaluator = cc_evaluate if dataset == 'Contra_Costa' else main_evaluate
    frozen = read_json(directory/'frozen.json')
    payload = {key: value for key, value in frozen.items() if key != 'frozen_sha256'}
    require(frozen.get('frozen_sha256') == digest_json(payload)
            and frozen.get('schema') == evaluator.SCHEMA
            and frozen.get('dataset') == dataset and frozen.get('split') == 'test'
            and frozen.get('engineering_only') is False
            and frozen.get('arms') == ['FULL'] and frozen.get('seeds') == [2025]
            and frozen.get('metric_contract') == evaluator.METRIC_CONTRACT
            and frozen.get('evaluation_source') == evaluator.evaluation_sources(),
            'Expected unchanged FULL-only completed test cache: '+str(directory))
    key = 's2025__FULL'
    require(set(frozen['records']) == {key}, 'Unexpected result records')
    role = frozen['records'][key]['role']
    require(role.get('completed_training_epochs') == 100
            and role.get('full_protocol_reached') is True
            and role.get('selection') == 'best'
            and role.get('selected_on') == 'validation_pooled_mae',
            'Incomplete training or non-validation selection')
    cached = evaluator.read_cache(directory/key, frozen, key)
    require(cached.get('test_evaluated') is True, 'Result is not a test evaluation')
    return cached, frozen['frozen_sha256']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args(argv)
    rows, sources = [], {}
    for dataset in ('Alameda', 'Contra_Costa', 'Orange'):
        record, checksum = read_result(args.run_root, dataset)
        metrics = record['native_metrics']
        row = dict(dataset=dataset, method='PASE-Net', seed=2025,
                   best_epoch=record['checkpoint_role']['epoch'],
                   parameters=record['parameters']['registered'])
        per_lead = {item['lead']: item for item in metrics['per_horizon']}
        for label, values in [('h3', per_lead[3]), ('h6', per_lead[6]),
                              ('h12', per_lead[12]), ('average', metrics['horizon_macro'])]:
            for metric in ('mae', 'rmse', 'mape_percent'):
                row[label+'_'+metric] = values[metric]
        rows.append(row)
        sources[dataset] = checksum
    with (args.run_root/'main_results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Current compact PASE-Net: three-dataset test results', '',
             'Seed 2025; 100-epoch budget; validation-selected BEST; FULL only.',
             'Contra Costa reuses its completed compact FULL. Alameda/Orange train from scratch.',
             '12-lead mean metrics; RMSE is the mean of per-lead RMSE, not pooled RMSE.', '',
             '| Dataset | BEST epoch | Parameters | Average MAE | Average RMSE | Average MAPE % |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} |'.format(
            row['dataset'], row['best_epoch'], row['parameters'],
            row['average_mae'], row['average_rmse'], row['average_mape_percent']))
    lines.extend(['', 'H3/H6/H12 and unrounded metrics: main_results.csv.',
                  'Existing public test splits are not a new independent holdout.', ''])
    (args.run_root/'main_report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(args.run_root/'main_results.json', dict(rows=rows, sources=sources,
        test_evaluated=True, new_training_or_forwards=False, independent_confirmation=False))
    print('PASE_COMPACT_MAIN_RESULTS_COMPLETE:', args.run_root/'main_report.md')


if __name__ == '__main__':
    main()
