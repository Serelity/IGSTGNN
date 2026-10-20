"""Read-only ORIGINAL_IGSTGNN validation prediction diagnosis. No model imports.

This is a retrospective development diagnosis, not independent confirmation.
It can find or fail to find the specified prediction pattern. It never fits,
calibrates, retrains, loads a checkpoint, or opens test/future input datasets.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'original_baseline_gw16_validation_v1'
EXPECTED_IDS = np.arange(3909, 4746, dtype=np.int64)
PATTERN_RULE = dict(region_change_skill_strictly_above=0.,
                    local_change_skill_at_most=0.,
                    local_mae_strictly_above_region_mae=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text(value):
    value = str(value or '').strip()
    return '' if value.lower() in ('', 'nan', 'none', 'unknown', 'na') else value


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def partition(rows):
    groups, regions = {}, []
    for i, row in enumerate(rows):
        groups.setdefault((text(row['Fwy']), text(row['Direction'])), []).append(i)
    for group, members in sorted(groups.items()):
        def key(i):
            mile = number(rows[i]['Abs PM'])
            return (not np.isfinite(mile), mile if np.isfinite(mile) else 0.,
                    int(rows[i]['station_id']))
        ordered = sorted(members, key=key)
        for j, chunk in enumerate(np.array_split(ordered, int(np.ceil(len(ordered) / 32)))):
            regions.append(dict(name='/'.join(group) + ':' + str(j),
                                members=[int(i) for i in chunk]))
    return regions


def read_original(directory, data_dir):
    directory, data_dir = Path(directory).resolve(), Path(data_dir).resolve()
    meta_file, archive = directory / 'metadata.json', directory / 'predictions.npz'
    require(meta_file.is_file() and archive.is_file(), 'Missing original metadata.json/predictions.npz')
    metadata = json.loads(meta_file.read_text(encoding='utf-8'))
    require(metadata.get('schema') == SCHEMA and metadata.get('status') == 'ORIGINAL_VALIDATION_COMPLETE'
            and metadata.get('test_evaluated') is False and metadata.get('engineering_only') is False
            and metadata.get('original_full_model') is True and metadata.get('native_pos') is False
            and metadata.get('registered_parameters') == 455220,
            'Requires complete, unmodified ORIGINAL_IGSTGNN validation replay; no test/probe/NATIVE_POS')
    require(metadata.get('native_replay', {}).get('passed') is True
            and metadata.get('source', {}).get('training_complete') is True, 'Incomplete original replay/training')
    hashes = metadata['data_identity']['hashes']
    require(digest(archive) == metadata['archive_sha256'], 'Prediction checksum changed')
    sensor_file = data_dir / 'sensors.csv'
    require(digest(sensor_file) == hashes['sensors.csv'], 'Sensor metadata differs from original release')
    with np.load(archive, allow_pickle=False) as payload:
        require(set(payload.files) == {'prediction', 'target', 'sample_ids', 'station_ids'}, 'Original fields differ')
        arrays = {key: payload[key] for key in payload.files}
    require(digest(archive) == metadata['archive_sha256'], 'Prediction changed during read')
    p, y = arrays['prediction'], arrays['target']
    require(p.shape == y.shape == (837, 12, 496, 1) and p.dtype == y.dtype == np.float32
            and np.isfinite(p).all() and np.isfinite(y).all(), 'Original values, dtype or dimensions differ')
    require(arrays['sample_ids'].dtype == np.int64 and np.array_equal(arrays['sample_ids'], EXPECTED_IDS),
            'Only original validation sample indices 3909..4745 accepted')
    require(np.array_equal(metadata['data_identity']['splits']['val']['selected_indices'], EXPECTED_IDS),
            'Original metadata validation sample axis differs')
    with sensor_file.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = np.asarray([int(row['station_id']) for row in rows], dtype=np.int64)
    require(len(ids) == len(set(ids.tolist())) == 496
            and arrays['station_ids'].dtype == np.int64 and np.array_equal(ids, arrays['station_ids'])
            and np.array_equal(ids, metadata['station_ids']), 'Station axis differs; no implicit reordering')
    return arrays, metadata, rows


def read_pase(directory, original, original_metadata, rows):
    directory = Path(directory).resolve()
    metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    role = metadata.get('checkpoint_role', {})
    require(metadata.get('schema') == 'position_state_gw16_prediction_archive_v1'
        and metadata.get('arm') == 'STATE_EXCHANGE' and metadata.get('split') == 'val'
        and metadata.get('engineering_only') is False and metadata.get('test_evaluated') is False
        and metadata.get('validation_replay', {}).get('passed') is True
        and role.get('selection') == 'best' and role.get('completed_training_epochs') == 100
        and role.get('full_protocol_reached') is True, 'Requires completed PASE validation BEST, not LAST/test/probe')
    file = directory/'predictions.npz'
    require(digest(file) == metadata['archive_sha256'], 'PASE cache checksum changed')
    with np.load(file, allow_pickle=False) as payload:
        arrays = {k: payload[k] for k in ('prediction', 'target', 'sample_ids', 'station_ids')}
    for key in ('target', 'sample_ids', 'station_ids'):
        require(arrays[key].dtype == original[key].dtype and np.array_equal(arrays[key], original[key]),
                'PASE and original must use identical values and axes: '+key)
    require(arrays['prediction'].shape == original['prediction'].shape
            and arrays['prediction'].dtype == np.float32 and np.isfinite(arrays['prediction']).all(),
            'Invalid PASE prediction shape/type/values')
    require(metadata['identity']['data']['hashes'] == original_metadata['data_identity']['hashes'],
            'PASE and original releases differ')
    data = metadata['identity']['data']
    require(np.array_equal(data['station_ids'], original['station_ids']), 'PASE identity station axis differs')
    expected_regions = partition(rows)
    region_identity = data['partition']
    require(region_identity['policy'] == 'road_direction_postmile' and region_identity['max_region_size'] == 32
            and region_identity['members'] == [r['members'] for r in expected_regions],
            'PASE must be the fixed road/direction/postmile partition, not a random-region variant')
    require(digest(file) == metadata['archive_sha256'], 'PASE cache changed during read')
    return arrays, metadata


def skill(error_square, truth_square):
    denominator = float(np.sum(truth_square))
    return None if denominator <= 1e-12 else float(1. - np.sum(error_square) / denominator)


def decompose(prediction, target):
    """One H x K fixed-node record. MAEs are separate, not additive."""
    p, y = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    require(p.shape == y.shape and p.ndim == 2 and p.shape[0] >= 2, 'Expected H x fixed nodes')
    require(np.isfinite(p).all() and np.isfinite(y).all(), 'Nonfinite record')
    pm, ym = p.mean(1), y.mean(1)
    pl, yl = p - pm[:, None], y - ym[:, None]
    error, regional_error, local_error = p - y, pm - ym, pl - yl
    rd, pd = np.diff(ym), np.diff(pm)
    ld, pld = np.diff(yl, axis=0), np.diff(pl, axis=0)
    local_energy = float(np.mean(local_error ** 2))
    region_energy = float(np.mean(regional_error ** 2))
    node_energy = float(np.mean(error ** 2))
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    result = dict(nodes=y.shape[1], node_mae=float(np.mean(abs(error))),
        region_mae=float(np.mean(abs(regional_error))), local_mae=float(np.mean(abs(local_error))),
        node_mse=node_energy, region_mse=region_energy, local_mse=local_energy,
        mse_identity_error=abs(node_energy-region_energy-local_energy),
        region_change_skill=skill((pd-rd)**2, rd**2),
        local_change_skill=skill((pld-ld)**2, ld**2),
        true_region_change_rms=rms(rd), true_local_change_rms=rms(ld),
        predicted_local_change_rms=rms(pld),
        local_change_error_square=float(np.sum((pld-ld)**2)),
        local_change_truth_square=float(np.sum(ld**2)),
        region_change_error_square=float(np.sum((pd-rd)**2)),
        region_change_truth_square=float(np.sum(rd**2)))
    result['pattern_observed'] = bool(result['region_change_skill'] is not None
        and result['region_change_skill'] > 0 and result['local_change_skill'] is not None
        and result['local_change_skill'] <= 0 and result['local_mae'] > result['region_mae'])
    return result, (pm, ym, pl, yl)


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_cache(root):
    # Known already-completed replay first; do not select by model accuracy.
    known = Path(root) / 'IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/original_native_val_20261019_223356'
    if (known/'predictions.npz').is_file():
        return known
    found = []
    for code in sorted(Path(root).glob('IGSTGNN*')):
        runs = code/'experiments/chronological_runs'
        if not runs.is_dir():
            continue
        for directory in sorted(runs.glob('original_native_val_*')):
            try:
                metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
                if metadata.get('schema') == SCHEMA and metadata.get('status') == 'ORIGINAL_VALIDATION_COMPLETE':
                    found.append((directory, metadata.get('archive_sha256')))
            except (OSError, ValueError):
                continue
    require(found, 'No completed original validation cache found; use --original-eval PATH. No inference was started.')
    require(len({h for _, h in found}) == 1,
            'Multiple different original caches found; specify one existing cache explicitly:\n'
            + '\n'.join(str(p) for p, _ in found))
    return found[0][0]


def plot_case(output, p, y, record, rows, valid, pase=None, pattern_summary=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42,
        'ps.fonttype':42, 'axes.linewidth':.7, 'savefig.facecolor':'white'})
    pm, ym = p.mean(1), y.mean(1)
    pl, yl = p-pm[:, None], y-ym[:, None]
    # Fixed low/mid/high TRUE local-change activity: no prediction-error maximization.
    ordering = np.argsort(np.mean(abs(np.diff(yl, axis=0)), axis=0), kind='stable')
    positions = sorted(set(int(round(q*(len(ordering)-1))) for q in (0., .5, 1.)))
    chosen = [int(ordering[i]) for i in positions]
    fig = plt.figure(figsize=(7.2, 4.7), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=(1., 1.15))
    upper = fig.add_subplot(grid[0, :2])
    note = fig.add_subplot(grid[0, 2]); note.axis('off')
    xx = np.arange(1, 13)*5
    def curves(ax, truth, pred, improved=None):
        ax.plot(xx, truth, color='#202327', lw=1.3, label='Ground truth')
        ax.plot(xx, pred, color='#A43855', ls='--', lw=1.3, label='IGSTGNN')
        ax.fill_between(xx, truth, pred, color='#F3C9D5', alpha=.65, label='IGSTGNN error')
        if improved is not None:
            ax.plot(xx, improved, color='#277DA8', lw=1.3, label='PASE')
        ax.set_xticks((5, 15, 30, 45, 60))
        ax.set_xlabel('Forecast lead (min)')
        ax.set_ylabel('Traffic flow (released units)')
        ax.tick_params(length=3, width=.6)
    curves(upper, ym, pm, None if pase is None else pase.mean(1))
    upper.set_title('(a) Regional average: identical fixed-node subset', loc='left', fontsize=9)
    upper.legend(frameon=False, ncol=3, loc='best', fontsize=7)
    def sf(v):
        return 'NA' if v is None else f'{v:.3f}'
    note.text(0, 1, 'Validation only\nSample '+str(record['sample_id'])+'\n'+record['region_name']
        +'\nFixed nodes: '+str(len(valid))+'\n\nIGSTGNN skills\nRegion change: '+sf(record['region_change_skill'])
        +'\nLocal change skill: '+sf(record['local_change_skill'])
        +'\n\n'+pattern_summary+'\nRetrospective diagnosis\nNot causal attribution', va='top', fontsize=7, linespacing=1.4)
    trace = []
    for col, j in enumerate(chosen):
        ax = fig.add_subplot(grid[1, col]); node = int(valid[j])
        curves(ax, y[:, j], p[:, j], None if pase is None else pase[:,j])
        ax.set_title(f'({chr(98+col)}) Station {rows[node]["station_id"]}\nPostmile {rows[node]["Abs PM"]}',
                     loc='left', fontsize=8)
        for h in range(12):
            trace.append(dict(sample_id=record['sample_id'], region=record['region_name'],
                station_id=rows[node]['station_id'], lead=h+1, lead_minutes=int(xx[h]),
                target=float(y[h,j]), prediction=float(p[h,j]), regional_target=float(ym[h]),
                regional_prediction=float(pm[h]), local_target=float(yl[h,j]),
                local_prediction=float(pl[h,j]),
                pase_prediction=None if pase is None else float(pase[h,j])))
    fig.savefig(output/'case_prediction_error.pdf', bbox_inches='tight')
    fig.savefig(output/'case_prediction_error.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    write_csv(output/'case_plotted_values.csv', trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/seu_share/home/huangkai/220243809/paper/IGSTGNN'))
    parser.add_argument('--original-eval', type=Path)
    parser.add_argument('--pase-eval', type=Path,
                        help='Completed STATE_EXCHANGE validation BEST cache; optional paired comparison')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args(argv)
    source = (args.original_eval or find_cache(args.root)).resolve()
    data_dir = (args.data_dir or args.root/'data/xtraffic/Contra_Costa').resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), 'Use a fresh diagnostic output; previous results are preserved')
    for base in (source, data_dir, *([args.pase_eval.resolve()] if args.pase_eval else [])):
        require(output != base and base not in output.parents and output not in base.parents,
                'Diagnostic output must be separate from data and source cache')
    for parent in (output, *output.parents):
        require(not (parent/'identity.json').is_file(), 'Output must be outside training run directories')
    arrays, metadata, rows = read_original(source, data_dir)
    p, y = arrays['prediction'][...,0].astype(np.float64), arrays['target'][...,0].astype(np.float64)
    improved, improved_meta = read_pase(args.pase_eval, arrays, metadata, rows) if args.pase_eval else (None, None)
    pp = improved['prediction'][...,0].astype(np.float64) if improved is not None else None
    regions, records, masks, per_lead = partition(rows), [], {}, []
    lead_sums = np.zeros((12, 5), dtype=np.float64)
    for s in range(len(p)):
        for r, region in enumerate(regions):
            members = np.asarray(region['members'], dtype=int)
            good = np.all(np.isfinite(y[s][:,members]) & (y[s][:,members] > .001), axis=0)
            valid = members[good]
            if len(valid) < 4 or len(valid) < int(np.ceil(.8*len(members))):
                continue
            record, values = decompose(p[s][:,valid], y[s][:,valid])
            if pp is not None:
                paired, _ = decompose(pp[s][:,valid], y[s][:,valid])
                record.update({'pase_'+k:v for k,v in paired.items()})
                record['pase_reduces_node_and_local_mae'] = bool(paired['node_mae'] < record['node_mae']
                                                               and paired['local_mae'] < record['local_mae'])
            record.update(sample_position=s, sample_id=int(arrays['sample_ids'][s]), region=r,
                          region_name=region['name'], coverage=len(valid)/len(members))
            records.append(record); masks[(s,r)] = valid
            pm, ym, pl, yl = values
            lead_sums[:,0] += abs(pm-ym)*len(valid)
            lead_sums[:,1] += np.sum(abs(pl-yl), axis=1)
            lead_sums[:,2] += np.sum(abs(p[s][:,valid]-y[s][:,valid]), axis=1)
            lead_sums[:,3] += len(valid)
            lead_sums[:,4] += 1
    require(records, 'No eligible fixed-node records')
    require(max(r['mse_identity_error'] for r in records) < 1e-8, 'MSE identity failed')
    for h, v in enumerate(lead_sums):
        per_lead.append(dict(lead=h+1, valid_node_cells=int(v[3]), window_regions=int(v[4]),
            region_absolute_error_node_weighted=float(v[0]/v[3]),
            local_deviation_mae=float(v[1]/v[3]), node_mae=float(v[2]/v[3])))
    if pp is not None:
        c_sums = np.zeros((12, 3), dtype=np.float64)
        for (s,r), valid in masks.items():
            pm, ym = pp[s][:,valid].mean(1), y[s][:,valid].mean(1)
            c_sums[:,0] += abs(pm-ym)*len(valid)
            c_sums[:,1] += np.sum(abs((pp[s][:,valid]-pm[:,None])-(y[s][:,valid]-ym[:,None])),axis=1)
            c_sums[:,2] += np.sum(abs(pp[s][:,valid]-y[s][:,valid]),axis=1)
        for h, row in enumerate(per_lead):
            row.update(pase_region_absolute_error_node_weighted=float(c_sums[h,0]/lead_sums[h,3]),
                       pase_local_deviation_mae=float(c_sums[h,1]/lead_sums[h,3]),
                       pase_node_mae=float(c_sums[h,2]/lead_sums[h,3]))
    patterns = [r for r in records if r['pattern_observed']]
    # Representative median true local activity within the defined pattern, never max error.
    pool = patterns or records
    median = float(np.median([r['true_local_change_rms'] for r in pool]))
    example = min(pool, key=lambda r:(abs(r['true_local_change_rms']-median), r['sample_id'], r['region']))
    report = dict(schema='igstgnn_local_problem_diagnosis_v1',
        source_original_validation=str(source), original_archive_sha256=metadata['archive_sha256'],
        script_sha256=digest(Path(__file__)), sensor_sha256=digest(data_dir/'sensors.csv'),
        scope='retrospective_reused_development_validation', test_evaluated=False,
        analysis_model_inference_calls=0, optimizer_updates=0, fitted_calibration=False,
        target_mask='new diagnostic subset, NOT native mask: same finite raw targets > .001 throughout 12 leads; >=80% region coverage and >=4 nodes',
        total_windows=837, fixed_regions=len(regions), potential_records=837*len(regions),
        eligible_records=len(records), excluded_records=837*len(regions)-len(records),
        pattern_rule=PATTERN_RULE, pattern_records=len(patterns), pattern_fraction=len(patterns)/len(records),
        pattern_windows=len(set(r['sample_id'] for r in patterns)),
        pattern_regions=len(set(r['region'] for r in patterns)),
        aggregate_region_change_skill=skill([r['region_change_error_square']*r['nodes'] for r in records],
                                           [r['region_change_truth_square']*r['nodes'] for r in records]),
        aggregate_local_change_skill=skill([r['local_change_error_square'] for r in records],
                                          [r['local_change_truth_square'] for r in records]),
        example=example, example_selection='nearest median true local-change RMS within pattern records, or ALL eligible records if pattern absent; tie sample then region',
        example_is_supporting_case=bool(patterns), per_horizon=per_lead,
        mechanism_validated=False, independent_confirmation=False, statistical_significance_claimed=False,
        cautions=['A change-space zero-increment diagnostic is not a deployed forecasting baseline',
          'High local error fraction and reduced predicted variance do not alone establish a model defect',
          'Unpredictable noise and rational conditional-mean shrinkage remain possible explanations',
          'Overlapping development windows are not independent events',
          'This pattern is descriptive and does not prove position causes errors or PASE exchange fixes them'])
    if pp is not None:
        total_cells = sum(r['nodes'] for r in records)
        paired_rows = []
        for label, prefix in (('ORIGINAL_IGSTGNN',''),('PASE','pase_')):
            paired_rows.append(dict(method=label, **{k:float(sum(r[prefix+k]*r['nodes'] for r in records)/total_cells)
                for k in ('node_mae','region_mae','local_mae')}))
        report['paired_comparison'] = dict(source_pase_validation=str(args.pase_eval.resolve()),
            pase_archive_sha256=improved_meta['archive_sha256'], pase_best=improved_meta['checkpoint_role'],
            same_targets_samples_stations_and_fixed_masks=True, rows=paired_rows,
            original_pattern_records_improved_on_both_node_and_local_mae=sum(r['pase_reduces_node_and_local_mae'] for r in patterns),
            all_records_improved_on_both_node_and_local_mae=sum(r['pase_reduces_node_and_local_mae'] for r in records),
            case_selection_did_not_use_pase_predictions=True,
            training_recipes_identical=False, exchange_mechanism_causally_identified=False)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'window_regions.csv', records)
    write_csv(output/'per_horizon.csv', per_lead)
    grouped = []
    for r, region in enumerate(regions):
        selected = [v for v in records if v['region'] == r]
        if selected:
            grouped.append(dict(region=region['name'], records=len(selected),
                pattern_records=sum(v['pattern_observed'] for v in selected),
                pattern_fraction=sum(v['pattern_observed'] for v in selected)/len(selected)))
    write_csv(output/'by_region.csv', grouped)
    if not args.no_figures:
        s, r = example['sample_position'], example['region']; valid = masks[(s,r)]
        label = f'Pattern records: {len(patterns)}/{len(records)}' if patterns else 'Pattern NOT OBSERVED\nNeutral example'
        plot_case(output, p[s][:,valid], y[s][:,valid], example, rows, valid,
                  None if pp is None else pp[s][:,valid], label)
    require(digest(source/'predictions.npz') == metadata['archive_sha256'], 'Source changed during diagnosis')
    if improved_meta is not None:
        require(digest(args.pase_eval/'predictions.npz') == improved_meta['archive_sha256'], 'PASE source changed during diagnosis')
    (output/'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    status = 'OBSERVED' if patterns else 'NOT_OBSERVED'
    lines = ['# IGSTGNN 实际预测误差：区域走势与局域变化诊断', '',
        '本诊断步骤只读已完成的BEST验证预测；前置.sh入口可分别从权重运行一次验证预测。不训练，不读取测试缓存。', '',
        '这是在当前改进之后，对重复使用的开发验证集所做的回顾诊断，不伪称事前发现或独立确认。', '',
        '## 本次诊断运行前固定的可观察模式（非研究事前预注册）', '',
        '同一窗口、同一固定区域节点：区域变化技能>0、局域变化技能<=0，且局域偏离MAE大于区域平均MAE。',
        '变化技能=1−预测相邻时距变化的平方误差/真实相邻时距变化的平方和。分母为零记NA。',
        '这里的零变化是变化空间中的诊断参照，不是可部署的原始流量基线；区域技能>0仅表示比此参照好，不直接等于区域预测准确。',
        '固定掩码是本次诊断子集，不替代原版原生指标；所有模型用同一子集比较。', '',
        f'模式是否观察到：**{status}**；{len(patterns)}/{len(records)}个合格窗口—区域（{100*len(patterns)/len(records):.2f}%）。',
        f'涉及{report["pattern_windows"]}/837个验证窗口、{report["pattern_regions"]}/{len(regions)}个固定区域。',
        f'全部潜在记录{report["potential_records"]}，因覆盖/区域大小过滤{report["excluded_records"]}；不把过滤零流量统一解释为缺失。', '',
        '## 结果如何用于论文', '',
        '若OBSERVED，图仅展示符合定义的真实预测模式；必须同时报告上述覆盖比例和per_horizon.csv，不能仅凭一个案例宣称普遍缺陷。',
        '若NOT_OBSERVED，示例从全部合格记录按中位规则取，只是普通误差图，不是该假设的支持证据。',
        '平方误差的区域/局域分解是恒等式，MAE不能相加；局域误差多或预测幅度小均不足以单独证明过平滑。',
        '不据此宣称差异都可预测、误差来自位置、双向交换有效或创新性成立。需已有对照/消融承担机制归因。', '',
        '## 文件', '',
        '- diagnosis.json：完整来源、判据、覆盖、样例与限制。',
        '- window_regions.csv、by_region.csv、per_horizon.csv：所有记录及分组，不筛掉反例。',
        ('- case_prediction_error.png / .pdf、case_plotted_values.csv：真实预测和真实值，采样点间直线连接。'
         if not args.no_figures else '- --no-figures：未导出图像或绘图CSV。'),
        '- 图中横轴是预测时距，不拼接重叠窗口，不虚构绝对时间或事故起始线。', '']
    if pp is not None:
        lines += ['## 同样本 BEST 预测直接比较', '',
            '| 方法 | 站点 MAE | 区域平均 MAE | 局域偏离 MAE |',
            '|---|---:|---:|---:|']
        for row in report['paired_comparison']['rows']:
            lines.append('| {} | {:.6f} | {:.6f} | {:.6f} |'.format(row['method'],row['node_mae'],row['region_mae'],row['local_mae']))
        lines += ['', '以上对比复用完全相同的目标、节点、样本和固定掩码。图中PASE为蓝实线，IGSTGNN为红虚线。',
            '样例仅按IGSTGNN表现和真实变化选择，没有按PASE提升最大挑选；PASE无改善或更差也照常展示。',
            '这可以观察整套模型是否改善同类预测偏差；训练配方也有差异，交换机制贡献由消融回答。', '']
        write_csv(output/'paired_overall.csv', report['paired_comparison']['rows'])
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status='DIAGNOSIS_COMPLETE', pattern_status=status, records=len(records),
                         pattern_records=len(patterns), output_dir=str(output)), ensure_ascii=False), flush=True)
    print('Read '+str(output/'report.md'), flush=True)
    return report


if __name__ == '__main__':
    main()
