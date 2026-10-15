"""Read small release inputs and NPY headers for the proposed DIF contract.

This does not deserialize incident_all.npy, inspect traffic/targets, construct a
model, replay a checkpoint, train, or evaluate any data split. Source contracts
and observed metadata are deliberately reported separately.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_header(path):
    # Only the header is read; pickle/object payloads are never deserialized.
    with path.open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported NPY header version: '+str(version))
        header_bytes = stream.tell()
    require(dtype.hasobject and len(shape) == 1, 'Expected one-dimensional object release')
    return dict(shape=list(shape), dtype=str(dtype), fortran_order=bool(fortran),
                header_bytes_read=header_bytes, file_bytes=path.stat().st_size,
                object_payload_deserialized=False, full_file_rehashed=False)


def source_contract():
    paths = ('src/models/igstgnn.py', 'src/utils/igstgnn_published.py',
             'src/utils/igstgnn_published_events.py',
             'experiments/chronological/train_b3_r_minimal_paper.py')
    tree = ast.parse((REPO/paths[0]).read_text(encoding='utf-8-sig'))
    defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'model_args' and node.func.attr == 'setdefault'
                and len(node.args) == 2 and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in ('use_pre', 'dy_graph', 'sta_graph')):
            defaults[node.args[0].value] = ast.literal_eval(node.args[1])
    require(set(defaults) == {'use_pre', 'dy_graph', 'sta_graph'}, 'Model defaults changed')
    builder = ast.parse((REPO/paths[-1]).read_text(encoding='utf-8-sig'))
    arguments = {}
    for node in ast.walk(builder):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'IGSTGNN':
            for keyword in node.keywords:
                if keyword.arg == 'model_args':
                    require(isinstance(keyword.value, ast.Call), 'Unexpected builder arguments')
                    for item in keyword.value.keywords:
                        if item.arg in ('layer', 'k_s', 'k_t', 'gap', 'num_hidden'):
                            arguments[item.arg] = ast.literal_eval(item.value)
    require(set(arguments) == {'layer', 'k_s', 'k_t', 'gap', 'num_hidden'}, 'Builder contract changed')
    # The builder explicitly requests doubletransition: two adjacency modalities.
    require("'doubletransition'" in (REPO/paths[-1]).read_text(encoding='utf-8-sig'),
            'Adjacency normalization changed')
    modalities = 2
    counts = dict(predefined=modalities*arguments['k_s']*int(defaults['use_pre']),
                  dynamic=modalities*arguments['k_s']*int(defaults['dy_graph']),
                  static=arguments['k_s']*int(defaults['sta_graph']), self_bank=1)
    return dict(scope='source_contract_not_model_instantiation', defaults=defaults,
                builder=arguments, support_banks=counts, total_support_banks=sum(counts.values()),
                source_sha256={name: sha256(REPO/name) for name in paths})


def audit(args):
    protocol = json.loads(args.protocol.read_text(encoding='utf-8-sig'))
    hashes = {}
    for name in ('adj_matrix.npy', 'sensors.csv', 'incident_stats.npz',
                 'type_mapping.json', 'desc_mapping.json'):
        actual = sha256(args.data_dir/name)
        require(actual == protocol['data_sha256'][name], 'Release fingerprint mismatch: '+name)
        hashes[name] = actual
    header = object_header(args.data_dir/'incident_all.npy')
    samples = header['shape'][0]
    counts = dict(total_samples=samples, train_samples=int(.70*samples), val_samples=int(.15*samples))
    counts['test_samples'] = samples-counts['train_samples']-counts['val_samples']
    require(all(protocol[key] == value for key, value in counts.items()), 'Header split counts mismatch')
    with (args.data_dir/'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        station_ids = [int(row['station_id']) for row in csv.DictReader(stream)]
    n = len(station_ids)
    require(n == protocol['stations'] and len(set(station_ids)) == n, 'Station metadata axis mismatch')
    graph_path = args.graph_dir/'graph.json'
    graph = json.loads(graph_path.read_text(encoding='utf-8-sig'))
    graph_hash = sha256(graph_path)
    require(graph_hash == protocol['graph_sha256'], 'Candidate graph fingerprint mismatch')
    require(graph['station_ids'] == station_ids, 'Candidate graph station metadata differs')
    vocabulary = {}
    for name, count in (('type_mapping.json', 6), ('desc_mapping.json', 29)):
        mapping = json.loads((args.data_dir/name).read_text(encoding='utf-8-sig'))
        require(len(mapping) == count and all(type(v) is int for v in mapping.values())
                and sorted(mapping.values()) == list(range(count)), 'Vocabulary mismatch: '+name)
        vocabulary[name] = dict(categories=count, unknown_index=count)
    with np.load(args.data_dir/'incident_stats.npz', allow_pickle=False) as stats:
        scaler = dict(normalized=bool(stats['normalized']), mean=float(stats['mean']), std=float(stats['std']))
    require(scaler['normalized'] and np.isfinite([scaler['mean'], scaler['std']]).all()
            and scaler['std'] > 0, 'Published scaler metadata invalid')
    adjacency = np.load(args.data_dir/'adj_matrix.npy', allow_pickle=False)
    require(adjacency.shape == (n, n) and np.isfinite(adjacency).all() and (adjacency >= 0).all(),
            'Adjacency shape/value contract mismatch')
    positive = adjacency > 0
    # int32 avoids overflow: at most n paths contribute to one two-step entry.
    positive_two = (positive.astype(np.int32)@positive.astype(np.int32)) > 0
    support = np.eye(n, dtype=bool) | positive | positive.T | positive_two | positive_two.T
    unseen, components = set(range(n)), []
    while unseen:
        stack, size = [unseen.pop()], 0
        while stack:
            node = stack.pop()
            size += 1
            neighbours = {int(j) for j in np.flatnonzero(support[node])} & unseen
            unseen -= neighbours
            stack.extend(neighbours)
        components.append(size)
    source = source_contract()
    require(source['builder']['k_s'] == 2, 'This hard-support audit is defined for original k_s=2')
    bank_positions = {}
    for order, mask in ((1, positive), (2, positive_two)):
        forward = mask.copy()
        np.fill_diagonal(forward, False)
        reverse = forward.T
        bank_positions['forward_order_'+str(order)] = int(forward.sum())
        bank_positions['reverse_order_'+str(order)] = int(reverse.sum())
        bank_positions['static_order_'+str(order)] = int((forward | reverse).sum())
    density = float(support.mean())
    directional_positions = sum(value for key, value in bank_positions.items()
                                if key.startswith(('forward_', 'reverse_')))
    static_positions = sum(value for key, value in bank_positions.items() if key.startswith('static_'))
    defaults = source['defaults']
    all_bank_positions = (n + directional_positions*(int(defaults['use_pre'])+int(defaults['dy_graph']))
                          + static_positions*int(defaults['sta_graph']))
    return dict(
        schema='min_conditional_propagation_input_audit_v1',
        status='SMALL_INPUT_AUDIT_PASS_NOT_SAMPLE_OR_MODEL_CERTIFICATION',
        engineering_only=True, primary_baseline='original_IGSTGNN',
        observed=dict(release_small_file_sha256=hashes, published_object_header=header,
                      counts_from_header_and_frozen_rounding=counts,
                      station_axis=dict(count=n, unique=n, matches_candidate_graph_metadata=True,
                                        traffic_internal_station_axis_independently_verified=False),
                      vocabulary=vocabulary, scaler_metadata=scaler,
                      candidate_graph_sha256=graph_hash,
                      adjacency=dict(shape=list(adjacency.shape), dtype=str(adjacency.dtype),
                                     positive_count=int(positive.sum()),
                                     diagonal_positive_count=int(np.diag(positive).sum()),
                                     symmetric_exact=bool(np.array_equal(adjacency, adjacency.T)))),
        source_contract=source,
        proposed_hard_support=dict(rule='I OR S OR S.T OR S_squared OR S_squared.T; S=(raw_A>0)',
                                   mixed_direction_two_step_paths_added=False,
                                   edge_positions_including_self=int(support.sum()), density=density,
                                   weak_component_sizes=sorted(components, reverse=True),
                                   per_family_order_positions_excluding_self=bank_positions,
                                   finite_algebraic_support_not_certified_physical_road_direction=True,
                                   dense_score_elements_first_layer_batch64=64*10*n*n*3,
                                   sparse_score_elements_first_layer_batch64=64*10*int(support.sum())*3,
                                   score_float32_bytes_sparse_first_layer=64*10*int(support.sum())*3*4,
                                   planned_support_bank_positions_total=all_bank_positions,
                                   all_bank_score_elements_first_layer_batch64=64*10*all_bank_positions*3,
                                   all_bank_score_float32_bytes_first_layer=64*10*all_bank_positions*3*4,
                                   memory_scope='union_and_all_bank_score_arithmetic_not_full_training_peak'),
        unknown=['per_sample_X_Y_relation_content_not_read',
                 'absolute_calendar_order_and_cross_split_overlap_not_independently_verified',
                 'relation_column_generation_formulas_and_physical_direction_not_certified',
                 'nominal_slot_continuity_and_absolute_X_to_Y_timing_not_independently_verified',
                 'first_report_field_availability_and_incident_age_or_clearance_not_certified',
                 'formal_F2_predictions_and_original_IGSTGNN_comparable_archive_not_replayed'],
        traffic_arrays_loaded=False, target_arrays_loaded=False, model_constructed=False,
        checkpoint_loaded=False, formal_inference_calls=0, optimizer_updates=0, test_evaluated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data/xtraffic/Contra_Costa')
    parser.add_argument('--graph-dir', type=Path, default=REPO/'experiments/chronological/internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=REPO/'experiments/chronological/min_component_fusion_v1.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output is not None:
        require(not args.output.exists() or args.overwrite, 'Output exists; choose a new path or use --overwrite')
    result = audit(args)
    result['audit_script_sha256'] = sha256(Path(__file__))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w' if args.overwrite else 'x', encoding='utf-8') as stream:
            stream.write(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
