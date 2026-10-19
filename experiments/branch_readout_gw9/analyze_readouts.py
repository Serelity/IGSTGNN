"""CPU-only paired readout analysis of the pinned, completed GW9 probe.

No model imports, inference, fitted correction, ensemble, oracle or training.
The failed joint gate is retained; local groups are exploratory descriptions.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.branch_readout_gw9 import readout_history as history
from experiments.branch_readout_gw9 import readout_metrics as metrics

REPO = Path(__file__).resolve().parents[2]
SCHEMA = 'gw9_cached_readout_analysis_v1'
CASES = ('FROZEN_R','SUM_CONTROL','BRANCH_READOUT')
PAIRS = (('SUM_vs_R','FROZEN_R','SUM_CONTROL'),
    ('BRANCH_vs_SUM','SUM_CONTROL','BRANCH_READOUT'),('BRANCH_vs_R','FROZEN_R','BRANCH_READOUT'))
FLOW_EDGES = (-float('inf'),0.,5.,20.,50.,100.,200.,314.,float('inf'))
SAMPLE_BLOCK = 128
SOURCE_FILES = ('experiments/branch_readout_gw9/analyze_readouts.py',
    'experiments/branch_readout_gw9/readout_auth.py',
    'experiments/branch_readout_gw9/readout_metrics.py',
    'experiments/branch_readout_gw9/readout_history.py',
    'experiments/branch_readout_gw9/run_analysis.sh','verify_gw9_analysis_bundle.py')


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def flatten(value, prefix=''):
    result = {}
    for key,item in value.items():
        name = prefix+key
        if isinstance(item,dict): result.update(flatten(item,name+'.'))
        elif isinstance(item,(list,tuple)): result[name] = json.dumps(item,allow_nan=False)
        else: result[name] = item
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def group_rows(result, group_type, group, **details):
    rows = []
    for case,record in result['cases'].items():
        rows.append(dict(record_kind='case',group_type=group_type,group=group,
            case=case,**details,**flatten(record)))
    for pair,record in result['comparisons'].items():
        rows.append(dict(record_kind='comparison',group_type=group_type,group=group,
            comparison=pair,**details,**flatten(record)))
    return rows


def flow_masks(values):
    for lo,hi in zip(FLOW_EDGES[:-1],FLOW_EDGES[1:]):
        label='[{},{} )'.format('-inf' if np.isneginf(lo) else f'{lo:g}',
            'inf' if np.isposinf(hi) else f'{hi:g}').replace(' ','')
        yield label,(values>=lo)&(values<hi)


def safe_corr(x,y):
    x,y = np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    keep = np.isfinite(x)&np.isfinite(y)
    if int(keep.sum())<2: return None
    x,y = x[keep],y[keep]; x=x-x.mean(); y=y-y.mean()
    denominator = float(np.sqrt(np.dot(x,x)*np.dot(y,y)))
    return float(np.clip(np.dot(x,y)/denominator,-1.,1.)) if denominator>0 else None


def partition_record(name, groups, global_result):
    rows = []
    for pair,_,_ in PAIRS:
        counts = sum(item['comparisons'][pair]['count'] for item in groups)
        expected = global_result['comparisons'][pair]['count']
        auth.require(counts==expected,'Group partition count differs: '+name)
        contributions = [item['comparisons'][pair]['global_mae_contribution'] or 0. for item in groups]
        positive = float(sum(max(value,0.) for value in contributions))
        negative = float(sum(max(-value,0.) for value in contributions))
        net = positive-negative; global_gain=global_result['comparisons'][pair]['mae_gain']
        auth.require(np.isclose(net,global_gain,atol=1e-10,rtol=1e-9),
            'Disjoint group contributions do not recover global gain: '+name)
        rows.append(dict(partition=name,comparison=pair,valid_count=counts,
            positive_group_contribution=positive,negative_group_contribution=negative,
            net_global_contribution=net,
            between_group_cancellation_ratio=1.-abs(net)/(positive+negative) if positive+negative>0 else None,
            interpretation='descriptive_disjoint_group_aggregation_no_selected_group_router'))
    return rows


def paired_views(rows):
    groups={}
    for row in rows:groups.setdefault(row['group'],[]).append(row)
    result=[]
    for label,items in groups.items():
        case=next(row for row in items if row.get('case')=='SUM_CONTROL')
        gain=next(row for row in items if row.get('comparison')=='SUM_vs_R')
        branch=next(row for row in items if row.get('comparison')=='BRANCH_vs_SUM')
        result.append(dict(group=label,valid_count=case['count'],SUM_mae=case['mae'],
            SUM_signed_bias=case['signed_bias'],SUM_vs_R_mae_gain=gain['mae_gain'],
            BRANCH_vs_SUM_mae_gain=branch['mae_gain'],
            BRANCH_vs_SUM_global_contribution=branch['global_mae_contribution']))
    return result


def build_analysis(archive, history_features=None, history_metadata=None):
    y,valid,predictions = auth.validate_archive(archive)
    total = int(valid.sum())
    global_result = metrics.analyze_group(predictions,y,valid,total_valid_count=total)
    tables = {'cases':[], 'comparisons':[], 'by_horizon':[], 'by_node':[], 'by_flow':[],
        'by_predicted_flow':[], 'by_road':[], 'by_support':[], 'by_sample_block':[],
        'corrections':[], 'partition_cancellation':[], 'history_features':[], 'by_history':[],
        'matched_history':[], 'matched_contrasts':[]}
    for case,record in global_result['cases'].items(): tables['cases'].append(dict(case=case,**flatten(record)))
    for pair,record in global_result['comparisons'].items(): tables['comparisons'].append(dict(comparison=pair,**flatten(record)))
    tables['corrections'].append(dict(scope='all_valid_points',**flatten(global_result['corrections'])))
    partitions = {}
    def add(category,label,index=None,mask=None,**details):
        if index is None:
            p,t,v = predictions,y,valid
        else:
            p,t,v = {name:value[index] for name,value in predictions.items()},y[index],valid[index]
        item = metrics.analyze_group(p,t,v,mask=mask,total_valid_count=total)
        tables[category].extend(group_rows(item,category,label,**details))
        return item
    groups=[]
    for h in range(y.shape[1]):
        groups.append(add('by_horizon',str(h+1),(slice(None),slice(h,h+1),slice(None)),horizon=h+1))
    partitions['horizon']=groups
    groups=[]
    for n,node in enumerate(archive['node_ids']):
        groups.append(add('by_node',str(int(node)),(slice(None),slice(None),slice(n,n+1)),
            node_id=int(node),road=str(archive['road_labels'][n])))
    partitions['node']=groups
    groups=[]
    for label,mask in flow_masks(y):
        groups.append(add('by_flow',label,mask=mask,group_input='future_truth_evaluation_only'))
    partitions['future_truth_flow']=groups
    groups=[]
    for label,mask in flow_masks(predictions['SUM_CONTROL']):
        groups.append(add('by_predicted_flow',label,mask=mask,group_input='SUM_prediction_available_at_origin'))
    partitions['SUM_predicted_flow']=groups
    groups=[]
    for road in np.unique(archive['road_labels']):
        indices=np.flatnonzero(archive['road_labels']==road)
        groups.append(add('by_road',str(road),(slice(None),slice(None),indices)))
    partitions['road']=groups
    direct=archive['direct_support'][:,None,:]
    partitions['direct_support']=[add('by_support','direct',mask=direct),
        add('by_support','outside_direct',mask=~direct)]
    groups=[]
    for start in range(0,len(y),SAMPLE_BLOCK):
        stop=min(start+SAMPLE_BLOCK,len(y))
        groups.append(add('by_sample_block',str(start//SAMPLE_BLOCK),
            (slice(start,stop),slice(None),slice(None)),
            first_sample_id=int(archive['sample_ids'][start]),last_sample_id=int(archive['sample_ids'][stop-1]),
            order='author_sample_order_calendar_not_independently_verified'))
    partitions['author_order_sample_blocks']=groups
    for name,items in partitions.items(): tables['partition_cancellation']+=partition_record(name,items,global_result)
    decomposition={}
    for name,left,right in (('SUM_minus_R','SUM_CONTROL','FROZEN_R'),
        ('BRANCH_minus_R','BRANCH_READOUT','FROZEN_R'),('BRANCH_minus_SUM','BRANCH_READOUT','SUM_CONTROL')):
        correction=predictions[left].astype(np.float64)-predictions[right].astype(np.float64)
        decomposition[name]=metrics.correction_decomposition(correction,valid)
        tables['corrections'].append(dict(scope=name,**flatten(decomposition[name])))
    history_result=dict(available=False,reason='NPZ forecasts do not contain historical inputs; use --data-dir for origin descriptors.')
    if history_features is not None:
        auth.require(history_metadata is not None,'Authenticated history metadata is required')
        auth.require(set(history_features)==set(history.FEATURES),'Historical descriptor fields differ')
        sum_error=predictions['SUM_CONTROL'].astype(np.float64)-y.astype(np.float64)
        sum_gain=np.abs(predictions['FROZEN_R'].astype(np.float64)-y.astype(np.float64))-np.abs(sum_error)
        for name,values in history_features.items():
            auth.require(values.shape==(len(y),y.shape[2]),'Historical descriptor axis differs: '+name)
            cuts=history_metadata['train_input_cutpoints'][name]
            labels=history.assign_quartiles(values,cuts['cutpoints'])
            for h in [None,*range(y.shape[1])]:
                target_valid=valid if h is None else valid[:,h:h+1]
                expanded=np.broadcast_to(values[:,None,:],target_valid.shape)
                error=sum_error if h is None else sum_error[:,h:h+1]
                gain=sum_gain if h is None else sum_gain[:,h:h+1]
                keep=target_valid & np.isfinite(expanded)
                x=expanded[keep]
                tables['history_features'].append(dict(feature=name,horizon='all' if h is None else h+1,
                    valid_count=int(keep.sum()),mean=float(x.mean()) if len(x) else None,
                    input_std=float(x.std()) if len(x) else None,
                    pearson_with_SUM_absolute_error=safe_corr(x,np.abs(error[keep])),
                    pearson_with_SUM_signed_error=safe_corr(x,error[keep]),
                    pearson_with_SUM_vs_R_point_gain=safe_corr(x,gain[keep]),
                    training_cutpoints=json.dumps(cuts['cutpoints']),
                    interpretation='development_association_not_out_of_sample_predictability'))
            quartile_groups=[]
            for level in (-1,0,1,2,3):
                label='unavailable' if level==-1 else 'Q'+str(level+1)
                quartile_groups.append(add('by_history',name+':'+label,mask=(labels==level)[:,None,:],
                    feature=name,quartile=label,cutpoints_source='train_histories_only'))
            tables['partition_cancellation']+=partition_record('history_'+name,quartile_groups,global_result)
            if name in history.MATCHED_FEATURES:
                for h in range(y.shape[1]):
                    index=(slice(None),slice(h,h+1),slice(None))
                    for flow_label,flow_mask in flow_masks(history_features['last_flow']):
                        cells={}
                        for level in (-1,0,1,2,3):
                            label='unavailable' if level==-1 else 'Q'+str(level+1)
                            cells[level]=add('matched_history',name+':'+label,index,
                                ((labels==level)&flow_mask)[:,None,:],feature=name,quartile=label,
                                horizon=h+1,origin_last_flow_group=flow_label,
                                interpretation='stratified_description_not_causal_adjustment_or_router')
                        low,high=cells[0],cells[3]
                        low_count=low['cases']['SUM_CONTROL']['count']; high_count=high['cases']['SUM_CONTROL']['count']
                        comparable=low_count>0 and high_count>0
                        tables['matched_contrasts'].append(dict(feature=name,horizon=h+1,
                            origin_last_flow_group=flow_label,Q1_valid_count=low_count,Q4_valid_count=high_count,
                            Q4_minus_Q1_SUM_mae=high['cases']['SUM_CONTROL']['mae']-low['cases']['SUM_CONTROL']['mae'] if comparable else None,
                            Q4_minus_Q1_SUM_signed_bias=high['cases']['SUM_CONTROL']['signed_bias']-low['cases']['SUM_CONTROL']['signed_bias'] if comparable else None,
                            Q4_minus_Q1_SUM_vs_R_gain=high['comparisons']['SUM_vs_R']['mae_gain']-low['comparisons']['SUM_vs_R']['mae_gain'] if comparable else None,
                            Q4_minus_Q1_BRANCH_vs_SUM_gain=high['comparisons']['BRANCH_vs_SUM']['mae_gain']-low['comparisons']['BRANCH_vs_SUM']['mae_gain'] if comparable else None,
                            interpretation='within_same_horizon_and_origin_flow_description_no_node_matching'))
        history_result=dict(available=True,metadata=history_metadata,
            matched_controls=['forecast_horizon','origin_last_flow_fixed_bins'],
            predictive_skill_validated=False,causal_claim=False,trained_models=0)
    node_views=paired_views(tables['by_node'])
    helpful=sorted([row for row in node_views if (row['BRANCH_vs_SUM_global_contribution'] or 0)>0],
        key=lambda row:row['BRANCH_vs_SUM_global_contribution'],reverse=True)[:5]
    harmful=sorted([row for row in node_views if (row['BRANCH_vs_SUM_global_contribution'] or 0)<0],
        key=lambda row:row['BRANCH_vs_SUM_global_contribution'])[:5]
    history_quartiles=[row for row in tables['by_history'] if row.get('case')=='SUM_CONTROL'
        and row['feature'] in history.MATCHED_FEATURES]
    views=dict(by_horizon=paired_views(tables['by_horizon']),by_flow=paired_views(tables['by_flow']),
        by_support=paired_views(tables['by_support']),by_road=paired_views(tables['by_road']),
        branch_between_group_cancellation=[row for row in tables['partition_cancellation']
            if row['comparison']=='BRANCH_vs_SUM'],branch_helpful_node_extremes=helpful,
        branch_harmful_node_extremes=harmful,
        pooled_history_correlations=[row for row in tables['history_features'] if row['horizon']=='all'],
        history_quartiles=history_quartiles,matched_history_stratum_counts=[])
    for name in history.MATCHED_FEATURES:
        comparable=[row for row in tables['matched_contrasts'] if row['feature']==name
            and row['Q4_minus_Q1_SUM_mae'] is not None]
        views['matched_history_stratum_counts'].append(dict(feature=name,comparable_strata=len(comparable),
            Q4_SUM_mae_higher=sum(row['Q4_minus_Q1_SUM_mae']>0 for row in comparable),
            Q4_SUM_mae_lower=sum(row['Q4_minus_Q1_SUM_mae']<0 for row in comparable),
            Q4_SUM_mae_equal=sum(row['Q4_minus_Q1_SUM_mae']==0 for row in comparable),
            interpretation='overlapping_strata_not_independent_replications_no_significance_test'))
    return dict(schema=SCHEMA,status='GW9_READOUT_ANALYSIS_COMPLETE',global_metrics=global_result,
        diagnostic_views=views,
        correction_decomposition=decomposition,history=history_result,
        validation_rows=len(y),horizons=y.shape[1],nodes=y.shape[2],valid_count=total,
        group_table_rows={name:len(rows) for name,rows in tables.items()},
        optimizer_updates=0,inference_calls=0,model_imports=False,test_evaluated=False,
        screening_recomputed=False,joint_eligibility_changed=False,candidate_for_confirmation=False,
        fits_or_oracles_applied=False,
        interpretation='posthoc_repeated_development_set_descriptions_not_independent_confirmation'),tables


def report_text(result):
    fmt=lambda value:'NA' if value is None else f'{value:.9f}'
    lines=['# GW9 读出修正与剩余误差分析','',
        '仅分析已认证预测缓存。零模型推理、零优化器更新；本轮 GW9 joint 筛查失败结论保持不变。','',
        '| Case | MAE | RMSE | MAPE % | Signed bias |','|---|---:|---:|---:|---:|']
    for name,row in result['global_metrics']['cases'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in
            (row['mae'],row['rmse'],None if row['mape_fraction'] is None else 100*row['mape_fraction'],row['signed_bias']))+' |')
    lines+=['','| Comparison | MAE gain | Relative gain % | Gross help | Gross harm | Point cancellation |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in result['global_metrics']['comparisons'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in (row['mae_gain'],
            None if row['relative_gain'] is None else 100*row['relative_gain'],row['gross_help'],row['gross_harm'],row['cancellation_ratio']))+' |')
    lines+=['','| Comparison | MSE change | Mean-only component | Centered component |',
        '|---|---:|---:|---:|']
    for name,row in result['global_metrics']['comparisons'].items():
        mse=row['mse_decomposition']
        lines.append('| '+name+' | '+' | '.join(fmt(mse[key]) for key in
            ('total_mse_change','mean_only_mse_change','centered_mse_change'))+' |')
    lines+=['','MSE change 为候选减基准，负值改善；均值分量与中心化分量相加等于总变化，未将均值作为校准量应用。']
    correction=result['global_metrics']['corrections']
    lines+=['','SUM−R 与 BRANCH−R 的修正相关系数：'+fmt(correction['common_correction_correlation'])+'。',
        'BRANCH−SUM 的 RMS / SUM−R 的 RMS：'+fmt(correction['branch_delta_to_sum_rms_ratio'])+'。',
        '相关系数为 NA 表示样本不足或修正为常量，不代表“不相关”。','',
        'MAE gain 为正表示候选改善；gross help 与 gross harm 均除全部有效点数，二者之差等于全局 MAE 改善。',
        '点级 cancellation 描述改善与恶化的抵消；partition_cancellation.csv 另报时距、节点等互斥分组之间的抵消。',
        'by_horizon / by_node / by_flow / by_road / by_support / by_sample_block.csv 均保存全部组，包括空组；局部优势不能直接转为部署路由。','',
        '| Correction | Global constant energy share | Horizon constant energy share | Within-horizon centered share |',
        '|---|---:|---:|---:|']
    for name,row in result['correction_decomposition'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(row[key]) for key in
            ('global_constant_energy_share','horizon_constant_energy_share','within_horizon_centered_energy_share'))+' |')
    lines+=['','Horizon constant share 已包含 Global constant share；二者不能重复相加。可加总的是 global + additional_horizon + within_horizon，详见 analysis.json。',
        '上述均值/能量分解只描述现有修正；没有把开发集均值应用到预测，也没有拟合平移、缩放、融合或 oracle。',
        '按真实流量分组使用未来真值，只能评价；按 SUM 预测流量分组虽为起点可用，仍不能据开发集局部收益声称已验证可预测性。',
        '128 样本块遵循作者数组顺序，未独立确认真实日历顺序，也不作为独立重复实验。','']
    for label,key in (('时距','by_horizon'),('真实流量（仅事后评价）','by_flow'),
            ('事故直接支持','by_support'),('道路标签','by_road')):
        lines+=['',label+'分解：','',
            '| Group | Valid | SUM MAE | SUM bias | SUM−R MAE gain | BRANCH−SUM MAE gain |',
            '|---|---:|---:|---:|---:|---:|']
        for row in result['diagnostic_views'][key]:
            lines.append('| '+row['group'].replace('|','/')+' | '+str(row['valid_count'])+' | '+
                ' | '.join(fmt(row[field]) for field in ('SUM_mae','SUM_signed_bias','SUM_vs_R_mae_gain','BRANCH_vs_SUM_mae_gain'))+' |')
    lines+=['','BRANCH−SUM 的互斥分组抵消：','',
        '| Partition | Positive group contribution | Negative group contribution | Between-group cancellation |',
        '|---|---:|---:|---:|']
    for row in result['diagnostic_views']['branch_between_group_cancellation']:
        lines.append('| '+row['partition']+' | '+' | '.join(fmt(row[field]) for field in
            ('positive_group_contribution','negative_group_contribution','between_group_cancellation_ratio'))+' |')
    lines+=['','BRANCH 相对 SUM 的节点极值仅用于定位，完整节点表见 by_node.csv；不是选择节点部署的规则。','',
        '| Node | Valid | BRANCH−SUM MAE gain | Global contribution |','|---|---:|---:|---:|']
    for row in (*result['diagnostic_views']['branch_helpful_node_extremes'],*result['diagnostic_views']['branch_harmful_node_extremes']):
        lines.append('| '+row['group']+' | '+str(row['valid_count'])+' | '+
            fmt(row['BRANCH_vs_SUM_mae_gain'])+' | '+fmt(row['BRANCH_vs_SUM_global_contribution'])+' |')
    if result['history']['available']:
        lines+=['已接入认证历史输入；history_features.csv 给出 SUM 剩余误差及配对收益与历史描述量的关联。',
            'by_history.csv 的四分位边界仅由训练段历史输入确定；matched_history.csv 在同一预测时距和起点最后流量区间内分组。',
            '这仍是开发集描述性关联，不是因果效应或独立预测技能；邻域矩阵行方向不命名为物理上游或传播时延。',
            '作者整体 object 数组会反序列化测试对象；仅访问 train/val 的 x_data，没有使用训练目标或测试样本。']
        lines+=['','历史特征与 SUM 剩余误差的未控制相关：','',
            '| Feature | Valid | Pearson absolute error | Pearson signed error |',
            '|---|---:|---:|---:|']
        for row in result['diagnostic_views']['pooled_history_correlations']:
            lines.append('| '+row['feature']+' | '+str(row['valid_count'])+' | '+
                fmt(row['pearson_with_SUM_absolute_error'])+' | '+fmt(row['pearson_with_SUM_signed_error'])+' |')
        lines+=['','| Feature:quartile | Valid | SUM MAE | SUM signed bias |','|---|---:|---:|---:|']
        for row in result['diagnostic_views']['history_quartiles']:
            lines.append('| '+row['group']+' | '+str(row['count'])+' | '+fmt(row['mae'])+' | '+fmt(row['signed_bias'])+' |')
        lines+=['','同一时距与起点流量区间内，Q4 相对 Q1 的 SUM MAE：','',
            '| Feature | Comparable strata | Higher | Lower | Equal |','|---|---:|---:|---:|---:|']
        for row in result['diagnostic_views']['matched_history_stratum_counts']:
            lines.append('| '+row['feature']+' | '+' | '.join(str(row[key]) for key in
                ('comparable_strata','Q4_SUM_mae_higher','Q4_SUM_mae_lower','Q4_SUM_mae_equal'))+' |')
        lines+=['','这些 strata 含重叠样本与时距，不是独立重复，符号计数不是显著性检验。详见 matched_contrasts.csv 的两端样本数和原始差值。']
    else: lines+=['未提供 --data-dir，跳过历史输入关联；仅凭预测 NPZ 不能检验历史变化或邻域分歧的可预测性。']
    provenance=result.get('provenance',{})
    if provenance:
        lines+=['','BEST/LAST 归档字节相同：'+str(provenance['best_last_archive_same_bytes'])+'；逐元素预测相同：'+
            str(provenance['best_last_predictions_equal'])+'。不据此推断检查点轮次或权重身份。',
            '源 probe identity：'+provenance['source_identity_sha256']+'。']
    return '\n'.join(lines)+'\n'


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path)
    args=parser.parse_args(argv)
    for name in ('probe_dir','output_dir','data_dir'):
        if getattr(args,name) is not None: setattr(args,name,getattr(args,name).resolve())
    return args


def main(argv=None):
    args=parse_args(argv)
    print(json.dumps(dict(status='GW9_READOUT_AUTHENTICATING',probe_dir=str(args.probe_dir))),flush=True)
    best,last,certificate=auth.load_authenticated_run(args.probe_dir,REPO)
    output=auth.require_output_outside_sources(args.output_dir,args.probe_dir,certificate)
    if args.data_dir is not None:
        auth.require(args.data_dir!=output and args.data_dir not in output.parents,
            'Analysis output must be outside the authenticated data directory')
    execution_source={name:auth.sha256(REPO/name) for name in SOURCE_FILES}
    output.mkdir(parents=True)
    try:
        print(json.dumps(dict(status='GW9_READOUT_PREPARING_HISTORY',enabled=args.data_dir is not None)),flush=True)
        features,history_metadata=(None,None) if args.data_dir is None else history.load_origin_features(
            args.data_dir,certificate['data_identity'])
        print(json.dumps(dict(status='GW9_READOUT_COMPUTING_PAIRED_GROUPS',model_inference_calls=0)),flush=True)
        result,tables=build_analysis(best,features,history_metadata)
        _,_,last_predictions=auth.validate_archive(last)
        same_predictions=all(np.array_equal(best['pred__'+name],last_predictions[name]) for name in CASES)
        result['provenance']=dict(certificate=certificate,
            source_identity_sha256=certificate['source_identity_sha256'],selection='best',
            best_last_archive_same_bytes=auth.sha256(Path(certificate['evaluation_directory'])/'best'/'paired_predictions.npz')==
                auth.sha256(Path(certificate['evaluation_directory'])/'last'/'paired_predictions.npz'),
            best_last_predictions_equal=bool(same_predictions),
            analysis_execution_source=execution_source,created_utc=datetime.now(timezone.utc).isoformat(),
            python=sys.version,numpy=str(np.__version__),platform=platform.platform())
        for name,rows in tables.items(): write_csv(output/(name+'.csv'),rows)
        auth.assert_unchanged(certificate)
        if history_metadata is not None: history.assert_unchanged(history_metadata)
        auth.require(all(auth.sha256(REPO/name)==digest for name,digest in execution_source.items()),
            'Analysis implementation changed during calculation')
        text=report_text(result)
        (output/'report.md').write_text(text,encoding='utf-8')
        # Publish the machine-readable completion record after all artifacts.
        write_json(output/'analysis.json',result)
        print(json.dumps(dict(status=result['status'],output_dir=str(output),
            history_features_available=result['history']['available'],optimizer_updates=0,
            inference_calls=0,joint_eligibility_changed=False)),flush=True)
        return result
    except BaseException as error:
        write_json(output/'failure.json',dict(status='GW9_READOUT_ANALYSIS_FAILED',
            error_type=type(error).__name__,error=str(error),optimizer_updates=0,inference_calls=0,test_evaluated=False))
        raise


if __name__=='__main__': main()
