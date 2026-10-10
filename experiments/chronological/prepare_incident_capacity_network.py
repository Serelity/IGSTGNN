"""Build M4.1 TRAIN-X metadata chains and conditional cutoff report sets."""
import argparse
from bisect import bisect_left
from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_incident_expansion import RAW_SHA256, nearest_candidate, sensor_groups
from experiments.chronological.prepare_context import spatial_features
from src.models.incident_capacity_fusion import IncidentCapacityBranch
from src.utils.capacity_fusion_inputs import OriginalCapacityInputs
from src.utils.capacity_network_inputs import (SCHEMA, REPORT_COLUMNS, CapacityNetworkInputs, InformationProfile,
    associate_reports, cutoff_collections, metadata_graph, numeric_source_groups)
from src.utils.incident_corridor import read_json, read_rows, require, sha256, write_json, write_rows


def extract_reports(raw_path, events, published, output):
    """Release a minimal attributed train-only metadata excerpt for server reuse."""
    output, raw_path = Path(output), Path(raw_path)
    require(not output.exists(), 'Report bundle already exists')
    require(sha256(raw_path) == RAW_SHA256, 'Raw v8 report source identity mismatch')
    times = sorted({datetime.fromisoformat(r['t0']) for r in events})
    trigger_ids = {r['incident_id'] for r in events}
    groups, counts, entities, conflicts = sensor_groups(published), Counter(), {}, set()
    with raw_path.open(encoding='utf-8-sig', newline='') as stream:
        source = csv.DictReader(stream, delimiter='\t')
        require(set(('incident_id', 'dt', 'Fwy', 'Freeway_direction', 'Abs PM', 'Latitude', 'Longitude')) <= set(source.fieldnames), 'Missing report location fields')
        for index, row in enumerate(source):
            counts['source_rows'] += 1
            try:
                issued = datetime.strptime(row['dt'], '%m/%d/%Y %H:%M:%S')
            except ValueError:
                counts['invalid_time_rows'] += 1
                continue
            k = bisect_left(times, issued)
            if k == len(times) or not 0 <= (times[k]-issued).total_seconds()/60 <= 60:
                continue
            match = nearest_candidate(row, groups)
            if row['incident_id'] not in trigger_ids and (match is None or match['nearest_sensor_km'] > 5
                                                         or not match['postmile_agrees_within_10_source_units']):
                continue
            try:
                from experiments.chronological.audit_incident_expansion import road_key
                road, direction = road_key(row['Fwy'], row['Freeway_direction'].upper())
                pm, lat, lon = [float(row[key]) for key in ('Abs PM', 'Latitude', 'Longitude')]
                require(np.isfinite([pm, lat, lon]).all() and -90 <= lat <= 90 and -180 <= lon <= 180, 'Invalid report location')
            except (ValueError, KeyError):
                counts['invalid_location_rows'] += 1
                continue
            sid = row['incident_id'].strip()
            if not sid:
                counts['missing_entity_id_rows'] += 1
                continue
            record = dict(source_row_index=index, incident_id=sid, report_time=issued.isoformat(),
                          road_number=road, direction=direction, postmile=pm, latitude=lat, longitude=lon)
            if sid in entities:
                counts['duplicate_entity_rows'] += 1
                if any(entities[sid][key] != record[key] for key in REPORT_COLUMNS[2:]):
                    conflicts.add(sid)
            else:
                entities[sid] = record
    selected = [r for sid, r in sorted(entities.items()) if sid not in conflicts]
    output.mkdir(parents=True)
    with (output/'train_locations.tsv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_COLUMNS, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected)
    manifest = dict(schema='incident_train_location_excerpt_v1', rows=len(selected), source_counts=dict(counts),
                    conflicting_entity_ids=sorted(conflicts), raw_sha256=RAW_SHA256,
                    train_manifest_identity=[(int(r['sample_index']), r['incident_id'], r['t0']) for r in events],
                    policy=dict(lookback_minutes=60, association_radius_km=1., catalog_nearest_station_radius_km=5.,
                                catalog_postmile_tolerance_source_units=10,
                                scope='source rows preceding original train cutoffs by at most 60 minutes; same road/direction within 5km of model station, plus original trigger IDs',
                                recorded_dt_and_location_at_first_report_assumed=True,
                                first_publication_time_available=False, source_versions_available=False,
                                duration_description_type_included=False),
                    license='CC BY-NC 4.0', license_url='https://creativecommons.org/licenses/by-nc/4.0/',
                    attribution='TraffiDent / XTraffic, Xiaochuan Gou et al., NeurIPS 2025',
                    source_url='https://www.kaggle.com/datasets/gpxlcj/xtraffic/versions/8',
                    file_sha256=sha256(output/'train_locations.tsv'))
    write_json(output/'manifest.json', manifest)
    return manifest


def prepare(original, report_bundle, output, device='cpu'):
    output = Path(output)
    partial = output.with_name(output.name+'.partial')
    require(not output.exists() and not partial.exists(), 'Use a fresh output directory')
    report_available = report_bundle is not None
    report_hashes = {}
    if report_available:
        report_bundle = Path(report_bundle)
        manifest = read_json(report_bundle/'manifest.json')
        require(manifest['schema'] == 'incident_train_location_excerpt_v1'
                and manifest['raw_sha256'] == RAW_SHA256
                and manifest['policy']['lookback_minutes'] == 60
                and manifest['policy']['association_radius_km'] == 1.
                and manifest['policy']['recorded_dt_and_location_at_first_report_assumed'] is True
                and manifest['policy']['duration_description_type_included'] is False, 'Unexpected report bundle policy')
        require(manifest['train_manifest_identity'] == [[int(r['sample_index']), r['incident_id'], r['t0']] for r in original.events], 'Report bundle train identity mismatch')
        require(sha256(report_bundle/'train_locations.tsv') == manifest['file_sha256'], 'Report bundle checksum mismatch')
        reports = read_rows(report_bundle/'train_locations.tsv', '\t')
        require(len(reports) == manifest['rows'] and all(tuple(r) == REPORT_COLUMNS for r in reports), 'Report whitelist/count mismatch')
        report_hashes = {'manifest.json': sha256(report_bundle/'manifest.json'), 'train_locations.tsv': sha256(report_bundle/'train_locations.tsv')}
    else:
        reports = []
        manifest = dict(conflicting_entity_ids=[], policy=dict(lookback_minutes=60, association_radius_km=1.,
                        recorded_dt_and_location_at_first_report_assumed=True, first_publication_time_available=False,
                        source_versions_available=False, duration_description_type_included=False,
                        scope='new report source not supplied; native trigger input preserved'))
    require(len({r['incident_id'] for r in reports}) == len(reports), 'Duplicate report IDs')
    by_id = {r['incident_id']: r for r in reports}
    public = {int(r['station_id']): r for r in original.published}
    ordered = [public[int(sid)] for sid in original.station_ids]
    # Replay all frozen trigger identities/locations; no change to native ICSF/TIID.
    for i, row in enumerate(original.events if report_available else []):
        report = by_id.get(row['incident_id'])
        require(report is not None and report['report_time'] == row['report_time'], 'Original trigger missing or time mismatch')
        expected = spatial_features(dict(freeway=int(report['road_number']), direction=report['direction'], postmile=float(report['postmile'])), ordered)
        require(np.array_equal(expected, original.trigger['distances'][i]), 'Original trigger spatial replay mismatch')
    aliases = numeric_source_groups(original.values, original.station_ids)
    conservative = metadata_graph(original.station_ids, original.published, original.raw, aliases)
    graph = metadata_graph(original.station_ids, original.published, original.raw, aliases, ramp_policy='explicit_lumped')
    reports = associate_reports(reports, graph, original.published)
    collections = cutoff_collections(original.events, reports)
    collections['sample_indices'] = [int(r['sample_index']) for r in original.events]
    representatives = {r['t0']: i for i, r in reversed(list(enumerate(original.events)))}
    available = []
    for stamp in collections['cutoffs']:
        values = np.asarray(original.values[representatives[stamp]])
        available.append((np.isfinite(values) & (values >= 0)).all((0, 2)))
    available = np.asarray(available, dtype=bool)
    # Same cutoff means identical auxiliary histories, even though native triggers can differ.
    for i, row in enumerate(original.events):
        j = representatives[row['t0']]
        if i != j:
            require(np.array_equal(original.values[i], original.values[j], equal_nan=True), 'Same-cutoff auxiliary histories disagree')
    mask = np.asarray(graph['operator_mask'])
    dynamic = available & mask[None]
    alias_ids = {sid for group in aliases for sid in group}
    stations = []
    for i, sid in enumerate(original.station_ids):
        row = public[int(sid)]
        stations.append(dict(station_id=int(sid), node_index=i, road=row['Fwy'], direction=row['Direction'],
                             complete_numeric_X_cutoff_fraction=float(available[:, i].mean()),
                             metadata_chain_eligible=bool(mask[i]),
                             conservative_chain_eligible=bool(conservative['operator_mask'][i]),
                             metadata_chain_numeric_cutoff_fraction=float(dynamic[:, i].mean()),
                             unresolved_numeric_alias=int(sid) in alias_ids,
                             independent_observation_entity_certified=False,
                             source_imputation_time_certified=False,
                             directed_connection_certified=False, online_report_version_certified=False,
                             jointly_certified=False, local_CTM_qualified=False,
                             pending='source imputation and entity audit; independent direction/topology evidence; report first-availability/version evidence'))
    roads = []
    for road, direction in sorted({(r['road'], r['direction']) for r in stations}):
        selected = [r for r in stations if (r['road'], r['direction']) == (road, direction)]
        roads.append(dict(road=road, direction=direction, stations=len(selected),
                          metadata_chain_nodes=sum(r['metadata_chain_eligible'] for r in selected),
                          numeric_90pct_metadata_chain_nodes=sum(r['metadata_chain_numeric_cutoff_fraction'] >= .9 for r in selected),
                          jointly_certified_nodes=0))
    n = len(original.station_ids)
    linked_per_cutoff = [sum(reports[i]['edge_index'] >= 0 for i in collections['report_indices'][a:b])
                         for a, b in zip(collections['offsets'], collections['offsets'][1:])]
    summary = dict(schema=SCHEMA, status='M41_NETWORK_INPUT_PACK_COMPLETE_EXPLORATORY',
                   main_training_ready=False, online_semantics_certified=False,
                   original_train_samples=len(original.events), nodes=n, distinct_cutoffs=len(collections['cutoffs']),
                   frozen_trigger_location_replays=len(original.events) if report_available else 0,
                   candidate_pairs=len(graph['candidate_pairs']),
                   metadata_clear_pairs=sum(r['metadata_clear'] for r in graph['candidate_pairs']),
                   metadata_chain_nodes=int(mask.sum()), metadata_chain_fraction=float(mask.mean()),
                   conservative_no_ramp_chain_nodes=sum(conservative['operator_mask']),
                   conservative_metadata_clear_pairs=sum(r['metadata_clear'] for r in conservative['candidate_pairs']),
                   metadata_chain_components=len(graph['admitted_chains']),
                   metadata_internal_edges=sum(r['kind'] == 'internal' for r in graph['edges']),
                   metadata_boundary_edges=sum(r['kind'] != 'internal' for r in graph['edges']),
                   mapped_ramp_sources=len({sid for r in graph['edges'] for sid in r['ramp_source_ids']}),
                   unmapped_ramp_sources=len(graph['unmapped_ramp_source_ids']),
                   numeric_90pct_metadata_chain_nodes=int((dynamic.mean(0) >= .9).sum()),
                   jointly_certified_nodes=0, majority_joint_gate_passed=False,
                   numeric_duplicate_groups=aliases,
                   report_entities=len(reports), associated_report_entities=sum(r['edge_index'] >= 0 for r in reports),
                   information_available=dict(new_reports=report_available, ramp_exchanges=any(r['ramp_source_ids'] for r in graph['edges'])),
                   report_memberships=len(collections['report_indices']),
                   cutoffs_with_multiple_reports=sum(b-a > 1 for a, b in zip(collections['offsets'], collections['offsets'][1:])),
                   cutoffs_with_linked_reports=sum(x > 0 for x in linked_per_cutoff),
                   max_reports_at_cutoff=int(max(np.diff(collections['offsets']), default=0)),
                   policy=manifest['policy'], source_report_conflicts=manifest['conflicting_entity_ids'],
                   primary_evaluation='unchanged original trigger-conditioned rows; 1/multiplicity weights saved for separate sensitivity only',
                   source_provenance=original.semantics,
                   original_inputs_sha256=original.fingerprints,
                   report_inputs_sha256=report_hashes,
                   traffic_Y_or_gap_read=False, validation_test_traffic_read=False, real_optimizer_updates=0,
                   sources_sha256={name: sha256(REPO/name) for name in
                       ('src/utils/capacity_network_inputs.py', 'experiments/chronological/prepare_incident_capacity_network.py')})
    partial.mkdir(parents=True)
    write_json(partial/'graph.json', graph)
    write_json(partial/'conservative_graph.json', conservative)
    write_json(partial/'cutoffs.json', collections)
    if reports:
        write_rows(partial/'reports.csv', reports)
    else:
        (partial/'reports.csv').write_text(','.join(REPORT_COLUMNS+('edge_index', 'distance_km', 'confidence'))+'\n', encoding='utf-8')
    write_rows(partial/'stations.csv', stations)
    write_rows(partial/'roads.csv', roads)
    np.save(partial/'numeric_cutoff_available.npy', available, allow_pickle=False)
    np.save(partial/'common_structure_mask.npy', mask & np.asarray(conservative['operator_mask']), allow_pickle=False)
    write_json(partial/'information_ablation.json', dict(
        status='DESIGN_FROZEN_NO_TRAINING_RESULTS', primary_scope='original 3604 train rows / 917 validation rows; all 496 stations',
        native_IGSTGNN_incident_path='unchanged in every group',
        profiles=[dict(name=InformationProfile(report, ramp).name, new_reports=report, ramp_exchanges=ramp)
                  for ramp in (False, True) for report in (False, True)],
        report_comparisons=['ramps_0_reports_1 vs ramps_0_reports_0', 'ramps_1_reports_1 vs ramps_1_reports_0'],
        report_estimand='new report information under fixed graph, coverage, initialization, histories and training budget',
        structure_comparisons=['ramps_1_reports_0 vs ramps_0_reports_0', 'ramps_1_reports_1 vs ramps_0_reports_1'],
        structure_estimand='ramp exchange assumptions plus changed graph/coverage; not pure information isolation',
        evaluation_masks=['all 496 original stations', 'common_structure_mask.npy', 'added structure nodes separately'],
        keep_original_target_mask_and_primary_denominator=True, paired_seed=11,
        condition='joint qualification pending; exploratory trials require explicit assumptions',
        actual_training_implemented=False))
    summary['outputs_sha256'] = {p.name: sha256(p) for p in sorted(partial.iterdir())}
    write_json(partial/'summary.json', summary)
    pack = CapacityNetworkInputs(partial, original, allow_exploratory=True)
    g, weights = pack.graph(device)
    # Actual original histories + actual report collections on draft graph; no targets/training.
    best = int(np.argmax(linked_per_cutoff))
    index = representatives[collections['cutoffs'][best]]
    batch = pack.batch(np.array([index]), device)
    torch.manual_seed(41)
    branch = IncidentCapacityBranch(g, weights).to(device).eval()
    with torch.inference_mode():
        on = branch(**batch['capacity_inputs'])
        off = branch(**batch['capacity_inputs'], incident_enabled=False)
    same = lambda name: float((on[name]-off[name]).abs().max().item())
    require(same('initial') == same('boundary_demand') == same('forecast_delta') == 0., 'Report bypass or initial fusion mismatch')
    summary['readback'] = dict(status='REAL_X_METADATA_HYPOTHESIS_ROLLOUT_PASS',
                               sample_index=int(original.events[index]['sample_index']),
                               report_count=int(batch['capacity_inputs']['reports']['present'].sum()),
                               report_supported_edges=int((on['report_association'] > 0).sum()),
                               condition_coefficient_abs_max=same('coefficients') if g.edges else 0.,
                               initial_on_off_abs_max=same('initial'), boundary_on_off_abs_max=same('boundary_demand'),
                               initial_fusion_abs_max=same('forecast_delta'),
                               cpu_or_cuda_device=str(device),
                               graph_scope='metadata_hypothesis_only_not_certified_real_topology')
    profiles = []
    for ramp in (False, True):
        for report in (False, True):
            profile = InformationProfile(report, ramp)
            reader = CapacityNetworkInputs(partial, original, allow_exploratory=True, profile=profile)
            data = reader.batch(np.array([index]), device)
            profile_graph, _ = reader.graph(device)
            profiles.append(dict(name=profile.name, information_status=reader.information_status,
                                 graph_edges=profile_graph.edges, nodes=int(profile_graph.operator_mask.sum()),
                                 report_slots=data['capacity_inputs']['reports']['weights'].shape[1]))
    summary['information_profile_readback'] = profiles
    write_json(partial/'summary.json', summary)
    partial.rename(output)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'history-dir', 'sensors'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--metadata-dir', type=Path, default=REPO/'experiments/chronological/physics_metadata')
    p.add_argument('--identity', type=Path, default=REPO/'experiments/chronological/incident_corridor_selection_v1.json')
    p.add_argument('--report-bundle', type=Path, default=REPO/'experiments/chronological/report_metadata')
    p.add_argument('--extract-from-raw', type=Path)
    p.add_argument('--without-new-reports', action='store_true', help='Do not supply the optional new-report source')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    original = OriginalCapacityInputs(a.data_dir, a.history_dir, a.sensors, a.metadata_dir, a.identity)
    try:
        require(not (a.without_new_reports and a.extract_from_raw), 'Cannot extract and omit reports together')
        if a.extract_from_raw:
            extract_reports(a.extract_from_raw, original.events, original.published, a.report_bundle)
        result = prepare(original, None if a.without_new_reports else a.report_bundle, a.output_dir, a.device)
    finally:
        original.close()
    import json
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('sha256')}, indent=2))


if __name__ == '__main__':
    main()
