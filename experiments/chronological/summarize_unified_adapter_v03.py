"""Read-only paired seed summaries from authenticated v03 epoch artifacts.

This checks internal run/checkpoint/source consistency, not independent proof
that an experiment was executed or that overlapping events generalize. A
formal *budget* summary never certifies baseline superiority or significance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.chronological.original_igstgnn_runtime import CONFIG, RELEASE, sha256, write_json
from experiments.chronological.original_igstgnn_source import MANIFEST_SHA256, verify_original_source
from experiments.chronological.train_unified_adapter_v03 import (
    EXECUTION_FILES, SCHEMA as TRAIN_SCHEMA, load_checkpoint, validate_saved, verify_initializations)
from experiments.chronological.unified_incident_adapter_v03 import ARMS, DEFAULT_ARMS


SCHEMA = 'unified_adapter_paired_seed_summary_v03'
EXPECTED_SEEDS = (2025, 2026, 2027)
COMPARISONS = (
    ('BOTH_vs_ORIGINAL', 'ORIGINAL', 'BOTH'),
    ('R_ONLY_vs_H_ONLY', 'H_ONLY', 'R_ONLY'),
    ('BOTH_vs_R_ONLY', 'R_ONLY', 'BOTH'),
    ('BOTH_vs_DIRECT', 'DIRECT', 'BOTH'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def recipe(identity):
    """Everything held fixed across seeds, except actual random initial tensors."""
    config = dict(identity['training_config'])
    config.pop('seed', None)
    runtime = dict(identity['runtime'])
    runtime.pop('seed', None)
    return dict(training_config=config, runtime=runtime,
        **{name: identity[name] for name in ('arms', 'adapter_config', 'data',
            'source_fingerprint', 'execution_source', 'curriculum', 'holdout_rows',
            'protocol', 'scheduler', 'early_stopping_enabled', 'from_scratch')})


def _source_checks(identity):
    declared = identity.get('execution_source', {})
    require(set(declared) == set(EXECUTION_FILES), 'Execution source manifest is incomplete or unexpected')
    for name in EXECUTION_FILES:
        source = (REPO/name).resolve()
        require(source.is_relative_to(REPO.resolve()), 'Source path leaves repository')
        require(source.is_file() and sha256(source) == declared[name], 'Execution source hash differs: '+name)


def load_run(directory):
    directory = Path(directory).resolve()
    identity_path, budget_path = directory/'identity.json', directory/'budget.json'
    identity_hash, budget_hash = sha256(identity_path), sha256(budget_path)
    identity, budget = read_json(identity_path), read_json(budget_path)
    require(identity.get('schema') == TRAIN_SCHEMA, 'Only v03 training identities are accepted')
    asserted = identity.get('identity_sha256')
    payload = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    recomputed = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    require(recomputed == asserted, 'Run identity checksum is inconsistent')
    _source_checks(identity)
    verify_original_source()
    require(identity.get('source_fingerprint') == MANIFEST_SHA256, 'Frozen original source identity differs')
    seed = identity['training_config']['seed']
    require(type(seed) is int and identity['runtime'].get('seed') == seed, 'Seed identities disagree')
    arms = identity['arms']
    require(isinstance(arms, list) and bool(arms) and len(set(arms)) == len(arms)
            and set(arms) <= set(ARMS), 'Registered arms must be unique known v03 arms')
    require(set(identity.get('initializations', {})) == set(arms), 'Initialization roles do not match registered arms')
    verify_initializations(identity['initializations'])
    split = identity['data']['splits']['train']
    partition = split['partition']
    require(isinstance(partition, list) and bool(partition) and all(type(v) is int and v > 0 for v in partition)
            and sum(partition) == split['count'], 'Training partition/count is inconsistent')
    require(type(budget.get('target_completed_epochs')) is int
            and 1 <= budget['target_completed_epochs'] <= CONFIG['max_epochs'], 'Invalid declared epoch budget')
    states, artifacts, absent = {}, {}, []
    for arm in arms:
        run = directory/arm
        checkpoint = run/'last.pt'
        if not checkpoint.is_file():
            absent.append(arm)
            continue
        before = sha256(checkpoint)
        state = load_checkpoint(checkpoint)
        validate_saved(state, identity, arm, run, len(partition))
        require(all(row[key] >= 0 for row in state['history'] for key in
                    ('validation_mae', 'validation_rmse', 'validation_mape_fraction')),
                'A validation error metric cannot be negative')
        require(state['completed_epoch'] <= budget['target_completed_epochs'], 'Checkpoint exceeds declared run budget')
        best = run/state['best_checkpoint_file'] if state['best_checkpoint_file'] else None
        artifacts[arm] = dict(last_file_sha256=before, model_tensor_sha256=state['model_tensor_sha256'],
            completed_epoch=state['completed_epoch'], best_epoch=state['best_epoch'],
            best_checkpoint_file=state['best_checkpoint_file'],
            best_checkpoint_file_sha256=sha256(best) if best else None,
            last_file_unchanged_during_read=sha256(checkpoint) == before)
        require(artifacts[arm]['last_file_unchanged_during_read'], 'Checkpoint changed while being summarized: '+arm)
        # Keep only small, validated history records, not optimizer/model tensors.
        states[arm] = dict(completed_epoch=state['completed_epoch'], history=state['history'])
        del state
    common = min((states.get(arm, {}).get('completed_epoch', 0) for arm in arms), default=0)
    for epoch in range(common):
        orders = [states[arm]['history'][epoch].get('train_order_sha256') for arm in arms]
        require(all(isinstance(value, str) and len(value) == 64 for value in orders)
                and len(set(orders)) == 1, 'Paired arm training order identity differs at epoch '+str(epoch+1))
    require(sha256(identity_path) == identity_hash and sha256(budget_path) == budget_hash,
            'Run identity or budget changed during read; retry after training finishes')
    return dict(directory=str(directory), seed=seed, identity=identity, budget=budget,
        states=states, absent_arms=absent, checkpoint_roles=artifacts,
        source_files=dict(identity_json_sha256=identity_hash, budget_json_sha256=budget_hash,
            summary_json_sha256=sha256(directory/'summary.json') if (directory/'summary.json').is_file() else None),
        summary_metrics_trusted=False)


def aggregate(values):
    values = list(map(float, values))
    return dict(values=values, count=len(values), mean=float(np.mean(values)) if values else None,
        sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        minimum=min(values) if values else None, maximum=max(values) if values else None,
        range=max(values)-min(values) if values else None,
        positive_count=sum(value > 0 for value in values),
        negative_count=sum(value < 0 for value in values), zero_count=sum(value == 0 for value in values))


def full_author_split(identity):
    data, holdout = identity['data'], identity['holdout_rows']
    splits = data['splits']
    fit_count = 3909-holdout
    fit = splits.get('train', {})
    val = splits.get('val', {})
    expected = (data.get('hashes') == RELEASE and fit.get('count') == fit_count
        and fit.get('selected_indices') == list(range(fit_count))
        and val.get('count') == 837 and val.get('selected_indices') == list(range(3909, 4746)))
    if holdout:
        cal = splits.get('calibration', {})
        expected = expected and cal.get('count') == holdout and cal.get('selected_indices') == list(range(fit_count, 3909))
    return bool(expected)


def summarize_runs(run_dirs):
    require(bool(run_dirs), 'At least one run directory is required')
    normalized = [str(Path(path).resolve()) for path in run_dirs]
    require(len(set(normalized)) == len(normalized), 'Duplicate run directory')
    runs = [load_run(path) for path in normalized]
    seeds = [run['seed'] for run in runs]
    require(len(set(seeds)) == len(seeds), 'Repeated seed is not an independent paired seed run')
    first = recipe(runs[0]['identity'])
    for run in runs[1:]:
        current = recipe(run['identity'])
        require(current == first, 'Runs differ beyond seed/initial tensors: '+', '.join(
            key for key in first if first[key] != current[key]))
    runs.sort(key=lambda run: run['seed'])
    # A missing registered arm makes the common budget zero. Never compare a
    # five-epoch winner with another seed's hundred-epoch winner.
    completed = [run['states'].get(arm, {}).get('completed_epoch', 0)
                 for run in runs for arm in run['identity']['arms']]
    common = min(completed, default=0)
    comparisons = {}
    for name, control, candidate in COMPARISONS:
        rows = []
        for run in runs:
            if not common or control not in run['states'] or candidate not in run['states']:
                continue
            a, b = (run['states'][arm]['history'][:common] for arm in (control, candidate))
            best_a, best_b = (min(records, key=lambda row: row['validation_mae']) for records in (a, b))
            row = dict(seed=run['seed'], compared_completed_epochs=common,
                best_epochs={control: best_a['epoch'], candidate: best_b['epoch']})
            for selection, left, right in (('independent_best', best_a, best_b), ('common_last', a[-1], b[-1])):
                row[selection] = {metric+'_gain': float(left['validation_'+metric]-right['validation_'+metric])
                    for metric in ('mae', 'rmse', 'mape_fraction')}
            rows.append(row)
        comparisons[name] = dict(control=control, candidate=candidate, paired=rows,
            **{selection: {metric+'_gain': aggregate([row[selection][metric+'_gain'] for row in rows])
                for metric in ('mae', 'rmse', 'mape_fraction')}
               for selection in ('independent_best', 'common_last')})
    reasons = []
    if set(seeds) != set(EXPECTED_SEEDS) or len(seeds) != 3:
        reasons.append('requires_exactly_preregistered_seeds_2025_2026_2027')
    if not set(DEFAULT_ARMS) <= set(first['arms']):
        reasons.append('required_default_information_and_direct_controls_absent')
    if any(run['identity'].get('engineering_only') is not False for run in runs):
        reasons.append('engineering_only_or_missing_engineering_declaration')
    if any(not full_author_split(run['identity']) for run in runs):
        reasons.append('full_authenticated_author_train_validation_partition_absent')
    if any(run['identity'].get('from_scratch') is not True for run in runs):
        reasons.append('from_scratch_declaration_absent')
    if any(run['absent_arms'] or run['budget']['target_completed_epochs'] != 100
            or any(state['completed_epoch'] != 100 for state in run['states'].values()) for run in runs):
        reasons.append('every_registered_arm_must_complete_the_same_100_epoch_budget')
    formal = not reasons
    status = ('V03_PAIRED_REPEATS_FORMAL_BUDGET_COMPLETE' if formal else
              'V03_PAIRED_REPEATS_INSUFFICIENT_SEEDS' if len(seeds) < 3 else
              'V03_PAIRED_REPEATS_DESCRIPTIVE_ONLY')
    return dict(schema=SCHEMA, status=status, generated_at=datetime.now(timezone.utc).isoformat(),
        producer_source_sha256=sha256(Path(__file__)),
        formal_budget_complete=formal, formal_ineligibility_reasons=reasons,
        expected_seeds=list(EXPECTED_SEEDS), observed_seeds=sorted(seeds), paired_seed_count=len(seeds),
        common_compared_completed_epochs=common, checkpoint_selection='earliest_minimum_validation_MAE_within_common_budget',
        comparisons=comparisons, shared_recipe=first,
        runs=[{key: value for key, value in run.items() if key not in ('identity', 'states')} for run in runs],
        no_seed_selection=True, negative_results_preserved=True,
        exceeds_baseline_confirmed=False, statistical_significance_claimed=False,
        independent_event_generalization_certified=False, test_evaluated=False,
        boundaries=['validation_already_used_for_design_and_checkpoint_selection',
            'seed_standard_deviation_is_descriptive_not_pointwise_significance',
            'common_seed_does_not_certify_independent_accidents_or_windows',
            'checks_authenticate_internal_artifact_consistency_not_independent_execution',
            'formal_status_certifies_budget_and_recipe_only_not_superiority_or_generalization'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output_dir.exists(), 'Use a fresh output directory')
    report = summarize_runs(args.run_dirs)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'summary.json', report)
    lines = ['# v03 配对种子汇总', '', '状态：'+report['status'],
        '正式预算完整：'+str(report['formal_budget_complete']),
        '共同比较完成轮数：'+str(report['common_compared_completed_epochs']), '',
        '| 比较 | 配对种子 | MAE 收益均值 | 种子样本标准差 | 最小值 | 最大值 | 正收益数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    def number(value):
        return '不可用' if value is None else '{:.9f}'.format(value)
    for name, comparison in report['comparisons'].items():
        result = comparison['independent_best']['mae_gain']
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(name,
            result['count'], number(result['mean']), number(result['sample_std']),
            number(result['minimum']), number(result['maximum']), result['positive_count']))
    lines.extend(['', '收益为对照误差减候选误差；保留负收益。JSON 同时保存每个种子的值、RMSE/MAPE 和共同末轮对照。',
        '少于三个预设种子、工程检查或短跑只能作描述。正式状态仅表示同配方与预算完整，不证明超过基线或独立泛化。',
        '未评价测试集，不把窗口或预测点视作独立重复，不计算点级显著性。'])
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], formal_budget_complete=report['formal_budget_complete'],
                          output_dir=str(args.output_dir.resolve())), ensure_ascii=False))
    return report


if __name__ == '__main__':
    main()
