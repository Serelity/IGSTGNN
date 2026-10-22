"""Export the completed legacy Concat-MLP BEST as an inference-only PASE v2 artifact.

No model forward, optimizer update, training identity, or resume state is created.
The original run and its authenticated validation cache remain unchanged.
"""
import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import uuid

from experiments.position_state_gw16.common import (
    REPO, atomic_save, digest_json, exclusive_run, load_state, read_json,
    require, sha256, tensor_tree_hash, write_json)
from experiments.pase_paper_experiments import evaluate as legacy_evaluate
from experiments.pase_paper_experiments import state as legacy_state
from experiments.state_exchange_multidata.common import load_config

SCHEMA = 'pase_mlp_v2_promoted_inference_v1'


def metadata_digest(value):
    return digest_json({key: item for key, item in value.items() if key != 'metadata_sha256'})


def verify_recorded_sources(recorded):
    """Check each recorded dependency; new v2 files do not invalidate old runs."""
    require(isinstance(recorded, dict) and recorded, 'Missing legacy source certificate')
    for name, expected in recorded.items():
        relative = Path(name)
        path = (REPO/relative).resolve()
        require(not relative.is_absolute() and REPO in path.parents,
                'Recorded source escapes the repository: '+str(name))
        require(path.is_file() and sha256(path) == expected,
                'Legacy recorded execution source changed: '+str(name))


def validate_native(metrics):
    require(isinstance(metrics, dict) and len(metrics.get('per_horizon', [])) == 12,
            'Legacy native metrics must contain twelve horizons')
    require([row['lead'] for row in metrics['per_horizon']] == list(range(1, 13)),
            'Legacy native metric horizon axis changed')
    for row in [metrics['horizon_macro'], *metrics['per_horizon']]:
        require(all(math.isfinite(row[key]) and row[key] >= 0
                    for key in ('mae', 'rmse', 'mape_percent')),
                'Nonfinite legacy validation metric')


def authenticate(source_run, source_evaluation):
    source_run, source_evaluation = Path(source_run).resolve(), Path(source_evaluation).resolve()
    identity = read_json(source_run/'identity.json')
    legacy_state.validate_identity(identity)
    require(identity['seed'] == 2025 and not identity['engineering_only']
            and identity['config'] == load_config('Contra_Costa'),
            'Only completed seed-2025 Contra Costa legacy MLP can be promoted')
    verify_recorded_sources(identity['execution_source'])
    weights, role, selection = legacy_evaluate.selected_best(source_run, identity, 'CONCAT_MLP')
    require(role['completed_training_epochs'] == 100 and role['full_protocol_reached'],
            'Legacy MLP must finish its fixed 100-epoch budget')
    runtime = read_json(source_run/'runtime.json')
    frozen = read_json(source_evaluation/'frozen.json')
    frozen_payload = {key: value for key, value in frozen.items() if key != 'frozen_sha256'}
    require(frozen.get('schema') == legacy_evaluate.SCHEMA
            and frozen.get('frozen_sha256') == digest_json(frozen_payload)
            and frozen.get('split') == 'val' and frozen.get('engineering_only') is False
            and frozen.get('metric_contract') == legacy_evaluate.METRIC_CONTRACT,
            'Source must be an authenticated complete validation evaluation')
    verify_recorded_sources(frozen['evaluation_source'])
    key = 's2025__CONCAT_MLP'
    expected = frozen['records'][key]
    require(expected['identity_sha256'] == identity['identity_sha256']
            and expected['role'] == role and expected['expected_validation'] == selection,
            'Legacy frozen cache identity, role or runtime differs')
    legacy_state.validate_runtime(runtime, frozen['runtime']['2025'])
    metadata = legacy_evaluate.read_cache(source_evaluation/key, frozen, key)
    require(metadata['identity_sha256'] == identity['identity_sha256']
            and metadata['config'] == identity['config'] and metadata['split'] == 'val'
            and metadata['test_evaluated'] is False
            and metadata['selection_metric_replay']['passed'] is True,
            'Legacy cache is not the selected validation replay')
    validate_native(metadata['native_metrics'])
    # numpy_metrics has no regional/reference diagnostics. Those diagnostics
    # are authenticated by the original replay certificate, not recomputed here.
    pooled_keys = set(metadata['common_mask_pooled']) & set(selection)
    require({'mae', 'rmse', 'mape_percent', 'signed_bias'} <= pooled_keys,
            'Legacy pooled validation certificate is incomplete')
    legacy_evaluate.replay_validation(metadata['common_mask_pooled'],
        {name:selection[name] for name in pooled_keys}, tolerance=1e-3)
    sample_count = identity['data']['loader_batching']['val_loader']['samples']
    require(metadata['audit']['samples'] == sample_count == 837,
            'Legacy validation sample axis is incomplete')
    axes = {name: metadata[name] for name in ('target_tensor_sha256',
            'sample_ids_sha256', 'station_ids_sha256')}
    # Every completed paired row must certify exactly the same validation axes.
    for record_key in frozen['records']:
        directory = source_evaluation/record_key
        require(directory.is_dir(), 'Legacy paired evaluation is incomplete: '+record_key)
        paired = legacy_evaluate.read_cache(directory, frozen, record_key)
        require(all(paired[name] == value for name, value in axes.items()),
                'Legacy paired cache has different targets/sample/station axes')
    last = load_state(source_run/'CONCAT_MLP'/'last.pt')
    certificate = dict(identity=identity, runtime=runtime, frozen=frozen,
        cached_metadata=metadata, history=last['history'],
        source_files={
            'identity.json': sha256(source_run/'identity.json'),
            'runtime.json': sha256(source_run/'runtime.json'),
            'CONCAT_MLP/last.pt': sha256(source_run/'CONCAT_MLP'/'last.pt'),
            'CONCAT_MLP/'+role['checkpoint_file']: role['checkpoint_sha256'],
            'evaluation/frozen.json': sha256(source_evaluation/'frozen.json'),
            'evaluation/'+key+'/metadata.json': sha256(source_evaluation/key/'metadata.json')})
    return weights, role, selection, metadata, certificate


def load_promoted(directory, expected_dataset=None):
    """Return (state_dict, external BEST role, certified metadata), never resume state."""
    directory = Path(directory).resolve()
    metadata = read_json(directory/'metadata.json')
    require(metadata.get('schema') == SCHEMA
            and metadata.get('metadata_sha256') == metadata_digest(metadata)
            and metadata.get('artifact_role') == 'inference_only_not_training_resume'
            and metadata.get('dataset') == 'Contra_Costa'
            and metadata.get('seed') == 2025 and metadata.get('source_arm') == 'CONCAT_MLP'
            and metadata.get('optimizer_included') is False
            and metadata.get('new_model_forwards') == metadata.get('new_optimizer_updates') == 0,
            'Invalid promoted inference metadata')
    require(expected_dataset is None or expected_dataset == metadata['dataset'],
            'Promoted FULL dataset differs')
    require(sha256(directory/'model.pt') == metadata['model_sha256']
            and sha256(directory/'source_certificate.json') == metadata['certificate_sha256'],
            'Promoted artifact checksum changed')
    saved = load_state(directory/'model.pt')
    require(set(saved) == {'schema', 'model_state', 'model_tensor_sha256', 'source_role'}
            and saved['schema'] == SCHEMA
            and saved['model_tensor_sha256'] == tensor_tree_hash(saved['model_state'])
            == metadata['model_tensor_sha256']
            and saved['source_role'] == metadata['source_role'],
            'Promoted model payload or tensors changed')
    certificate = read_json(directory/'source_certificate.json')
    legacy_state.validate_identity(certificate['identity'])
    require(certificate['identity']['identity_sha256'] == metadata['source_identity_sha256']
            and certificate['cached_metadata']['native_metrics'] == metadata['native_metrics']
            and certificate['cached_metadata']['common_mask_pooled'] == metadata['common_mask_pooled']
            and certificate['runtime'] == metadata['runtime']
            and certificate['identity']['config'] == metadata['config']
            and certificate['identity']['data'] == metadata['data_identity']
            and certificate['cached_metadata']['checkpoint_role'] == metadata['source_role']
            and certificate['cached_metadata'] == metadata['cached_metadata']
            and certificate['cached_metadata']['parameters'] == metadata['parameters'],
            'Promoted source certificate differs')
    role = dict(metadata['source_role'], role_kind='external_promoted_inference',
        source_arm='CONCAT_MLP', model_sha256=metadata['model_sha256'],
        metadata_sha256=sha256(directory/'metadata.json'),
        metadata_payload_sha256=metadata['metadata_sha256'])
    require(role['completed_training_epochs'] == 100 and role['full_protocol_reached'] is True
            and role['best_includes_stage_zero'] is True
            and role['selected_on'] == 'validation_pooled_mae',
            'Promoted source did not complete the fixed validation-selected budget')
    validate_native(metadata['native_metrics'])
    return saved['model_state'], role, metadata


def reference_identity(directory):
    _, role, metadata = load_promoted(directory, 'Contra_Costa')
    return dict(directory=str(Path(directory).resolve()), **role,
        dataset=metadata['dataset'], source_identity_sha256=metadata['source_identity_sha256'])


def promote(source_run, source_evaluation, output_dir):
    source_run, source_evaluation, output_dir = map(lambda p: Path(p).resolve(),
                                                   (source_run, source_evaluation, output_dir))
    for source in (source_run, source_evaluation):
        require(output_dir != source and source not in output_dir.parents
                and output_dir not in source.parents,
                'Promotion output must not intersect the immutable source run or evaluation')
    with ExitStack() as locks:
        locks.enter_context(exclusive_run(source_run))
        locks.enter_context(exclusive_run(source_evaluation))
        weights, role, selection, cached, certificate = authenticate(source_run, source_evaluation)
        if output_dir.exists():
            _, existing_role, metadata = load_promoted(output_dir, 'Contra_Costa')
            require(metadata['source_identity_sha256'] == certificate['identity']['identity_sha256']
                    and metadata['source_role'] == role
                    and metadata['model_tensor_sha256'] == tensor_tree_hash(weights),
                    'Existing output belongs to a different promotion; keep it immutable')
            return metadata
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_dir.with_name('.'+output_dir.name+'.writing_'+uuid.uuid4().hex)
        temporary.mkdir()
        atomic_save(temporary/'model.pt', dict(schema=SCHEMA, model_state=weights,
            model_tensor_sha256=tensor_tree_hash(weights), source_role=role))
        write_json(temporary/'source_certificate.json', certificate)
        metadata = dict(schema=SCHEMA, dataset='Contra_Costa', seed=2025,
            artifact_role='inference_only_not_training_resume', source_arm='CONCAT_MLP',
            model_sha256=sha256(temporary/'model.pt'), model_tensor_sha256=tensor_tree_hash(weights),
            certificate_sha256=sha256(temporary/'source_certificate.json'),
            source_identity_sha256=certificate['identity']['identity_sha256'], source_role=role,
            source_run=str(source_run), source_evaluation=str(source_evaluation),
            config=certificate['identity']['config'], parameters=cached['parameters'],
            data_identity=certificate['identity']['data'],
            selection_metrics=selection, common_mask_pooled=cached['common_mask_pooled'],
            native_metrics=cached['native_metrics'], runtime=certificate['runtime'],
            cached_metadata=cached,
            validation_axes={name:cached[name] for name in ('target_tensor_sha256',
                'sample_ids_sha256', 'station_ids_sha256')},
            optimizer_included=False, training_identity_created=False, resumable=False,
            new_model_forwards=0, new_optimizer_updates=0,
            original_tensor_bits_preserved=True, test_evaluated=False,
            role='existing_validation_selected_model_not_new_training_budget')
        metadata['metadata_sha256'] = metadata_digest(metadata)
        write_json(temporary/'metadata.json', metadata)
        load_promoted(temporary, 'Contra_Costa')
        temporary.rename(output_dir)
        return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--source-evaluation', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    metadata = promote(args.source_run, args.source_evaluation, args.output_dir)
    print('PASE_MLP_V2_PROMOTION_COMPLETE: '+str(args.output_dir.resolve()))
    print('Inference-only; exact legacy tensor bits; zero new forwards/updates; not a resume checkpoint')
    return metadata


if __name__ == '__main__':
    main()
