"""Compact the authenticated v2 position-zero BEST without training or forwards.

The source validation metrics are reference values, not a prediction cache for
the changed matrix multiplication. The fixed compact BEST must be replayed on
validation by the v3 evaluator before its result can be reported.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import uuid

from experiments.position_state_gw16.common import (
    REPO, atomic_save, digest_json, exclusive_run, load_state, read_json,
    require, sha256, tensor_tree_hash, write_json)
from experiments.pase_mlp_v2 import evaluate as source_evaluate
from experiments.pase_mlp_v2 import state as source_state
from experiments.pase_mlp_v2.promote import validate_native, verify_recorded_sources
from experiments.state_exchange_multidata.common import load_config
from .model import compact_state_dict


SCHEMA = 'pase_compact_v3_promoted_inference_v1'
SOURCE_ARM = 'NO_POSITION_INPUT'
FIRST_WEIGHT = 'head.exchange.network.0.weight'


def metadata_digest(value):
    return digest_json({key: item for key, item in value.items() if key != 'metadata_sha256'})


def compact_parameters(source, width, weights):
    """Account for columns really removed from the registered FP32 model."""
    first = weights[FIRST_WEIGHT]
    require(first.ndim == 2 and first.shape[1] == 3*width and first.element_size() == 4,
            'Compact scorer must have three FP32 state blocks')
    removed = int(first.shape[0])*width
    needed = ('registered', 'trainable', 'structurally_active', 'coefficient_registered',
              'coefficient_active', 'inactive_registered', 'parameter_bytes', 'buffer_bytes')
    require(all(name in source for name in needed)
            and source['inactive_registered'] == removed
            and source['coefficient_registered']-source['coefficient_active'] == removed
            and source['registered']-source['structurally_active'] == removed,
            'Source position-zero parameter certificate is inconsistent')
    result = dict(source)
    for name in ('registered', 'trainable', 'coefficient_registered'):
        result[name] -= removed
    result['parameter_bytes'] -= removed*first.element_size()
    result['inactive_registered'] = 0
    result['active_rule'] = 'compact_MLP_shared_hidden_partial_output_rows_no_explicit_p_columns'
    result['one_way_unused_coefficient_rows_retained'] = False
    return result


def authenticate(source_run, source_evaluation):
    """Authenticate the original completed run and selected validation cache."""
    source_run, source_evaluation = Path(source_run).resolve(), Path(source_evaluation).resolve()
    identity = read_json(source_run/'identity.json')
    source_state.validate_identity(identity)
    require(identity['seed'] == 2025 and identity['dataset'] == 'Contra_Costa'
            and identity['engineering_only'] is False
            and identity['config'] == load_config('Contra_Costa'),
            'Only completed seed-2025 Contra Costa v2 position-zero BEST can be compacted')
    verify_recorded_sources(identity['execution_source'])
    weights, role, selection = source_evaluate.selected_best(source_run, identity, SOURCE_ARM)
    require(role['completed_training_epochs'] == 100 and role['full_protocol_reached'] is True,
            'Source position-zero arm must finish its fixed 100-epoch budget')
    runtime = read_json(source_run/'runtime.json')
    frozen = read_json(source_evaluation/'frozen.json')
    payload = {key: value for key, value in frozen.items() if key != 'frozen_sha256'}
    require(frozen.get('schema') == source_evaluate.SCHEMA
            and frozen.get('frozen_sha256') == digest_json(payload)
            and frozen.get('dataset') == 'Contra_Costa' and frozen.get('seeds') == [2025]
            and frozen.get('split') == 'val' and frozen.get('engineering_only') is False
            and frozen.get('metric_contract') == source_evaluate.METRIC_CONTRACT,
            'Source must be an authenticated complete v2 validation evaluation')
    verify_recorded_sources(frozen['evaluation_source'])
    key = 's2025__'+SOURCE_ARM
    require(key in frozen['records'], 'Source evaluation has no position-zero BEST record')
    expected = frozen['records'][key]
    require(expected['identity_sha256'] == identity['identity_sha256']
            and expected['role'] == role and expected['expected_validation'] == selection,
            'Source frozen cache identity or BEST role differs')
    source_state.validate_runtime(runtime, frozen['runtime']['2025'])
    cached = source_evaluate.read_cache(source_evaluation/key, frozen, key)
    require(cached['identity_sha256'] == identity['identity_sha256']
            and cached['config'] == identity['config'] and cached['split'] == 'val'
            and cached['test_evaluated'] is False
            and cached['selection_metric_replay']['passed'] is True,
            'Source cache is not the selected validation replay')
    validate_native(cached['native_metrics'])
    pooled_keys = set(cached['common_mask_pooled']) & set(selection)
    require({'mae', 'rmse', 'mape_percent', 'signed_bias'} <= pooled_keys,
            'Source pooled validation certificate is incomplete')
    source_evaluate.replay_validation(cached['common_mask_pooled'],
        {name: selection[name] for name in pooled_keys}, tolerance=1e-3)
    samples = identity['data']['loader_batching']['val_loader']['samples']
    require(cached['audit']['samples'] == samples == 837,
            'Source validation sample axis is incomplete')
    axes = {name: cached[name] for name in ('target_tensor_sha256',
            'sample_ids_sha256', 'station_ids_sha256')}
    for record_key in frozen['records']:
        directory = source_evaluation/record_key
        require(directory.is_dir(), 'Source paired evaluation is incomplete: '+record_key)
        paired = source_evaluate.read_cache(directory, frozen, record_key)
        require(all(paired[name] == value for name, value in axes.items()),
                'Source paired cache has different target/sample/station axes')
    last = load_state(source_run/SOURCE_ARM/'last.pt')
    certificate = dict(identity=identity, runtime=runtime, frozen=frozen,
        cached_metadata=cached, history=last['history'], source_files={
            'identity.json': sha256(source_run/'identity.json'),
            'runtime.json': sha256(source_run/'runtime.json'),
            SOURCE_ARM+'/last.pt': sha256(source_run/SOURCE_ARM/'last.pt'),
            SOURCE_ARM+'/'+role['checkpoint_file']: role['checkpoint_sha256'],
            'evaluation/frozen.json': sha256(source_evaluation/'frozen.json'),
            'evaluation/'+key+'/metadata.json': sha256(source_evaluation/key/'metadata.json')})
    return weights, role, selection, cached, certificate


def load_promoted(directory, expected_dataset=None):
    """Return compact tensors, an external BEST role, and source provenance."""
    directory = Path(directory).resolve()
    metadata = read_json(directory/'metadata.json')
    require(metadata.get('schema') == SCHEMA
            and metadata.get('metadata_sha256') == metadata_digest(metadata)
            and metadata.get('artifact_role') == 'inference_only_not_training_resume'
            and metadata.get('dataset') == 'Contra_Costa' and metadata.get('seed') == 2025
            and metadata.get('source_arm') == SOURCE_ARM
            and metadata.get('optimizer_included') is False
            and metadata.get('conversion_validation_required') is True
            and metadata.get('source_metrics_are_reference_only') is True
            and metadata.get('copied_prediction_cache') is False
            and metadata.get('new_model_forwards') == metadata.get('new_optimizer_updates') == 0,
            'Invalid compact inference metadata')
    require(expected_dataset is None or expected_dataset == metadata['dataset'],
            'Compact FULL dataset differs')
    require(sha256(directory/'model.pt') == metadata['model_sha256']
            and sha256(directory/'source_certificate.json') == metadata['certificate_sha256'],
            'Compact artifact checksum changed')
    saved = load_state(directory/'model.pt')
    require(set(saved) == {'schema', 'model_state', 'model_tensor_sha256', 'source_role'}
            and saved['schema'] == SCHEMA
            and saved['model_tensor_sha256'] == tensor_tree_hash(saved['model_state'])
            == metadata['model_tensor_sha256']
            and saved['source_role'] == metadata['source_role'],
            'Compact model payload or tensors changed')
    certificate = read_json(directory/'source_certificate.json')
    identity, cached = certificate['identity'], certificate['cached_metadata']
    source_state.validate_identity(identity)
    require(identity['identity_sha256'] == metadata['source_identity_sha256']
            and identity['config'] == metadata['config']
            and identity['data'] == metadata['data_identity']
            and certificate['runtime'] == metadata['runtime']
            and cached == metadata['cached_metadata']
            and cached['checkpoint_role'] == metadata['source_role']
            and cached['native_metrics'] == metadata['native_metrics']
            and cached['common_mask_pooled'] == metadata['common_mask_pooled'],
            'Compact source certificate differs')
    source_role = metadata['source_role']
    key = 's2025__'+SOURCE_ARM
    frozen = certificate['frozen']
    require(frozen.get('schema') == source_evaluate.SCHEMA
            and frozen.get('frozen_sha256') == digest_json(
                {name: value for name, value in frozen.items() if name != 'frozen_sha256'})
            and frozen['split'] == 'val' and frozen['records'][key]['role'] == source_role
            and frozen['records'][key]['expected_validation'] == metadata['selection_metrics']
            and cached.get('payload_sha256') == source_evaluate.cache_digest(cached)
            and cached['frozen_sha256'] == frozen['frozen_sha256']
            and all(metadata['validation_axes'][name] == cached[name]
                    for name in ('target_tensor_sha256', 'sample_ids_sha256', 'station_ids_sha256')),
            'Compact source validation selection or axes differ')
    width = int(metadata['config']['hidden'])
    first = saved['model_state'][FIRST_WEIGHT]
    require(metadata['parameters'] == compact_parameters(cached['parameters'], width, saved['model_state'])
            and metadata['conversion']['first_weight'] == FIRST_WEIGHT
            and metadata['conversion']['source_shape'] == [first.shape[0], 4*width]
            and metadata['conversion']['compact_shape'] == list(first.shape)
            and metadata['conversion']['retained_columns_sha256'] == tensor_tree_hash(first)
            and metadata['conversion']['removed_parameters'] == first.shape[0]*width,
            'Compact conversion or parameter accounting differs')
    role = dict(source_role, role_kind='external_compact_inference', source_arm=SOURCE_ARM,
        source_model_tensor_sha256=source_role['model_tensor_sha256'],
        model_tensor_sha256=metadata['model_tensor_sha256'],
        model_sha256=metadata['model_sha256'], metadata_sha256=sha256(directory/'metadata.json'),
        metadata_payload_sha256=metadata['metadata_sha256'])
    require(role['completed_training_epochs'] == 100 and role['full_protocol_reached'] is True
            and role['best_includes_stage_zero'] is True and role['selection'] == 'best'
            and role['selected_on'] == 'validation_pooled_mae',
            'Compact source did not complete its validation-selected fixed budget')
    validate_native(metadata['native_metrics'])
    return saved['model_state'], role, metadata


def reference_identity(directory):
    _, role, metadata = load_promoted(directory, 'Contra_Costa')
    return dict(directory=str(Path(directory).resolve()), **role,
        dataset=metadata['dataset'], source_identity_sha256=metadata['source_identity_sha256'])


def write_report(directory, metadata):
    source = metadata['native_metrics']['horizon_macro']
    lines = ['# PASE-Net 精简版：已有位置置零候选的权重转换', '',
        '来源：Contra Costa，seed2025，已完成100轮的 NO_POSITION_INPUT 验证BEST。',
        '本次移除交换器第一层的显式位置输入列；其余权重和模型结构保持对应。',
        '新增训练更新与模型前向次数均为0；model.pt 是推理权重，不是续跑检查点。',
        '输入矩阵宽度改变可能造成FP32累加差异，因此旧预测缓存没有复制为新结果。',
        '运行精简版评价入口完成固定BEST验证回放后，才能报告本版本的实际指标。', '',
        '| 来源BEST轮数 | 已训练轮数 | 来源验证MAE | 来源验证RMSE | 来源验证MAPE % |',
        '|---:|---:|---:|---:|---:|',
        '| {} | {} | {:.6f} | {:.6f} | {:.6f} |'.format(
            metadata['source_role']['epoch'], metadata['source_role']['completed_training_epochs'],
            source['mae'], source['rmse'], source['mape_percent']), '',
        '上表仅为来源指标，不是新的精简模型评价结果。',
        '删除参数：'+str(metadata['conversion']['removed_parameters'])+'；精简后注册参数：'
            +str(metadata['parameters']['registered'])+'。',
        '来源身份SHA256：'+metadata['source_identity_sha256'],
        '精简张量SHA256：'+metadata['model_tensor_sha256'], '']
    (Path(directory)/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def promote(source_run, source_evaluation, output_dir):
    source_run, source_evaluation, output_dir = map(lambda path: Path(path).resolve(),
        (source_run, source_evaluation, output_dir))
    for source in (source_run, source_evaluation):
        require(output_dir != source and source not in output_dir.parents
                and output_dir not in source.parents,
                'Promotion output must not intersect the immutable source run or evaluation')
    with ExitStack() as locks:
        locks.enter_context(exclusive_run(source_run))
        locks.enter_context(exclusive_run(source_evaluation))
        weights, role, selection, cached, certificate = authenticate(source_run, source_evaluation)
        width = int(certificate['identity']['config']['hidden'])
        compact = compact_state_dict(weights, width)
        compact_hash = tensor_tree_hash(compact)
        if output_dir.exists():
            _, _, metadata = load_promoted(output_dir, 'Contra_Costa')
            require(metadata['source_identity_sha256'] == certificate['identity']['identity_sha256']
                    and metadata['source_role'] == role
                    and metadata['model_tensor_sha256'] == compact_hash,
                    'Existing output belongs to a different conversion; keep it immutable')
            if not (output_dir/'report.md').is_file():
                write_report(output_dir, metadata)
            return metadata
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_dir.with_name('.'+output_dir.name+'.writing_'+uuid.uuid4().hex)
        temporary.mkdir()
        atomic_save(temporary/'model.pt', dict(schema=SCHEMA, model_state=compact,
            model_tensor_sha256=compact_hash, source_role=role))
        write_json(temporary/'source_certificate.json', certificate)
        first = compact[FIRST_WEIGHT]
        metadata = dict(schema=SCHEMA, dataset='Contra_Costa', seed=2025,
            artifact_role='inference_only_not_training_resume', source_arm=SOURCE_ARM,
            model_sha256=sha256(temporary/'model.pt'), model_tensor_sha256=compact_hash,
            certificate_sha256=sha256(temporary/'source_certificate.json'),
            source_identity_sha256=certificate['identity']['identity_sha256'], source_role=role,
            source_run=str(source_run), source_evaluation=str(source_evaluation),
            config=certificate['identity']['config'],
            parameters=compact_parameters(cached['parameters'], width, compact),
            data_identity=certificate['identity']['data'], selection_metrics=selection,
            common_mask_pooled=cached['common_mask_pooled'], native_metrics=cached['native_metrics'],
            runtime=certificate['runtime'], cached_metadata=cached,
            validation_axes={name: cached[name] for name in ('target_tensor_sha256',
                'sample_ids_sha256', 'station_ids_sha256')},
            conversion=dict(first_weight=FIRST_WEIGHT, source_shape=list(weights[FIRST_WEIGHT].shape),
                compact_shape=list(first.shape), removed_input_block='explicit_position_first_block',
                removed_parameters=first.shape[0]*width,
                source_columns_sha256=tensor_tree_hash(weights[FIRST_WEIGHT]),
                retained_columns_sha256=tensor_tree_hash(first),
                retained_tensors_preserve_bits=True, mathematically_equivalent=True,
                bitwise_prediction_equivalence_claimed=False),
            conversion_validation_required=True, source_metrics_are_reference_only=True,
            copied_prediction_cache=False, optimizer_included=False,
            training_identity_created=False, resumable=False,
            new_model_forwards=0, new_optimizer_updates=0, test_evaluated=False,
            role='existing_validation_selected_candidate_compacted_without_new_training_budget')
        metadata['metadata_sha256'] = metadata_digest(metadata)
        write_json(temporary/'metadata.json', metadata)
        load_promoted(temporary, 'Contra_Costa')
        write_report(temporary, metadata)
        temporary.rename(output_dir)
        return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--source-evaluation', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    metadata = promote(args.source_run, args.source_evaluation, args.output_dir)
    print('PASE_COMPACT_V3_PROMOTION_COMPLETE: '+str(args.output_dir.resolve()))
    print('Inference-only; zero new forwards/updates; fixed-BEST compact validation replay required')
    return metadata


if __name__ == '__main__':
    main()
