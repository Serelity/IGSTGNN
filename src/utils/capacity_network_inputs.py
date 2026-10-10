"""M4.1 metadata graph and cutoff report collections, explicitly exploratory.

No edge here is certified by station sorting. Known ramps can be explicit
lumped open exchanges; unknown connectors split chains. Targets stay sealed.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.chronological.audit_incident_expansion import road_key
from src.models.incident_capacity_exchange import ExchangeGraph
from src.utils.incident_corridor import read_json, read_rows, require, sha256, unique_index


SCHEMA = 'incident_capacity_network_m41_v1'
REPORT_COLUMNS = ('source_row_index', 'incident_id', 'report_time', 'road_number',
                  'direction', 'postmile', 'latitude', 'longitude')


@dataclass(frozen=True)
class InformationProfile:
    """Independent optional information switches; native incident inputs persist."""
    new_reports: bool = True
    ramp_exchanges: bool = True

    def __post_init__(self):
        require(type(self.new_reports) is bool and type(self.ramp_exchanges) is bool, 'Information switches must be boolean')

    @property
    def name(self):
        return f'ramps_{int(self.ramp_exchanges)}_reports_{int(self.new_reports)}'


def numeric_source_groups(values, station_ids, batch_size=128):
    """Exact original TRAIN-X stream aliases, not proof of physical identity."""
    digests = [hashlib.sha256() for _ in station_ids]
    for start in range(0, len(values), batch_size):
        batch = np.asarray(values[start:start+batch_size])
        for i, digest in enumerate(digests):
            digest.update(np.ascontiguousarray(batch[:, :, i, :]).tobytes())
    groups = defaultdict(list)
    for sid, digest in zip(station_ids, digests):
        groups[digest.hexdigest()].append(int(sid))
    return [ids for ids in groups.values() if len(ids) > 1]


def metadata_graph(station_ids, published, raw, aliases=(), *, ramp_policy='cut'):
    """Conservative draft chains from full metadata; no topology certification."""
    require(ramp_policy in ('cut', 'explicit_lumped'), 'Unknown ramp policy')
    public, full = unique_index(published), unique_index(raw)
    ids = list(map(int, station_ids))
    require(len(set(ids)) == len(ids) and set(ids) == set(public), 'Invalid station axis')
    position = {sid: i for i, sid in enumerate(ids)}
    alias_ids = {sid for group in aliases for sid in group}
    groups = defaultdict(list)
    for sid in ids:
        p, r = public[sid], full.get(sid)
        require(r is not None and p['Type'] == r['Type'] == 'Mainline', 'Missing mainline source')
        require(p['Fwy'] == r['Fwy Name'] and p['Direction'] == r['Direction']
                and p['County'] == r['County'], 'Road source mismatch')
        road_key(p['Fwy'], p['Direction'])
        for field in ('Abs PM', 'Lat', 'Lng'):
            a, b = float(p[field]), float(r[field])
            require(np.isfinite([a, b]).all() and abs(a-b) <= 1e-6, 'Metadata coordinate mismatch')
        groups[p['Fwy'], p['Direction']].append(sid)
    pairs, paths = [], []
    for (road, direction), group in sorted(groups.items()):
        reverse = direction in ('S', 'W')  # Declared assumption, not independent direction evidence.
        ordered = sorted(group, key=lambda sid: (float(public[sid]['Abs PM']), sid), reverse=reverse)
        source = [r for r in raw if r['Fwy Name'] == road and r['Direction'] == direction]
        require(all(np.isfinite(float(r['Abs PM'])) for r in source), 'Invalid source postmile')
        chain = [ordered[0]]
        for a, b in zip(ordered, ordered[1:]):
            lo, hi = sorted((float(public[a]['Abs PM']), float(public[b]['Abs PM'])))
            between = [r for r in source if lo-1e-6 <= float(r['Abs PM']) <= hi+1e-6
                       and int(r['station_id']) not in (a, b)]
            reasons = []
            if hi-lo <= 1e-6:
                reasons.append('coincident_postmile')
            allowed = ('Mainline', 'On Ramp', 'Off Ramp') if ramp_policy == 'explicit_lumped' else ('Mainline',)
            if any(r['Type'] not in allowed for r in between):
                reasons.append('ramp_or_connector_in_closed_interval' if ramp_policy == 'cut' else 'unknown_connector_in_closed_interval')
            if any(r['Type'] == 'Mainline' for r in between):
                reasons.append('additional_source_mainline_in_closed_interval')
            if a in alias_ids or b in alias_ids:
                reasons.append('duplicate_numeric_source_entity_unresolved')
            if all(abs(float(public[a][k])-float(public[b][k])) <= 1e-6 for k in ('Lat', 'Lng')):
                reasons.append('coincident_coordinates')
            pairs.append(dict(source_station=a, destination_station=b, road=road, direction=direction,
                              source_postmile=float(public[a]['Abs PM']), destination_postmile=float(public[b]['Abs PM']),
                              intervening_source_ids=[int(r['station_id']) for r in between],
                              break_reasons=reasons, metadata_clear=not reasons,
                              direct_connection_certified=False, physical_length_km=None))
            if reasons:
                paths.append(chain)
                chain = [b]
            else:
                chain.append(b)
        paths.append(chain)
    admitted = [path for path in paths if len(path) >= 3]
    edges, mask, boundary = [], [False]*len(ids), [False]*len(ids)
    for path_index, path in enumerate(admitted):
        first, last = path[0], path[-1]
        for sid in path:
            mask[position[sid]] = True
        boundary[position[first]] = boundary[position[last]] = True
        for a, b in [(-1, first), *zip(path, path[1:]), (last, -1)]:
            anchor = public[b if a == -1 else a]
            edges.append(dict(index=len(edges), source=-1 if a == -1 else position[a],
                              destination=-1 if b == -1 else position[b], source_station=a, destination_station=b,
                              kind='inlet' if a == -1 else ('exit' if b == -1 else 'internal'),
                              road=anchor['Fwy'], direction=anchor['Direction'], path_index=path_index,
                              weight=0. if a == -1 else 1.,
                              ramp_source_ids=[],
                              boundary_basis='observed_network_extent_or_metadata_connection_gap' if -1 in (a, b) else None))
    if ramp_policy == 'explicit_lumped':
        # The ramp lies between observed stations; its attachment is a stated
        # coarse-grid assumption, not a reconstructed ramp/cell geometry.
        for ramp in raw:
            if ramp['Type'] not in ('On Ramp', 'Off Ramp'):
                continue
            matches = []
            for edge in edges:
                if edge['kind'] != 'internal' or ramp['Fwy Name'] != edge['road'] or ramp['Direction'] != edge['direction']:
                    continue
                pa = float(public[edge['source_station']]['Abs PM'])
                pb = float(public[edge['destination_station']]['Abs PM'])
                pm = float(ramp['Abs PM'])
                if (pa <= pm < pb) if pa < pb else (pb < pm <= pa):
                    matches.append(edge)
            require(len(matches) <= 1, 'Ambiguous ramp attachment')
            inlet = ramp['Type'] == 'On Ramp'
            if matches:
                edge = matches[0]
                # At an observed source station, attach to that station itself.
                at_source = abs(float(ramp['Abs PM'])-float(public[edge['source_station']]['Abs PM'])) <= 1e-6
                node = edge['source'] if at_source or not inlet else edge['destination']
            else:
                exact = [i for i, sid in enumerate(ids) if mask[i]
                         and public[sid]['Fwy'] == ramp['Fwy Name'] and public[sid]['Direction'] == ramp['Direction']
                         and abs(float(public[sid]['Abs PM'])-float(ramp['Abs PM'])) <= 1e-6]
                if len(exact) != 1:
                    continue
                node = exact[0]
                edge = next(r for r in edges if node in (r['source'], r['destination']))
            pair = (-1, node) if inlet else (node, -1)
            existing = next((r for r in edges if (r['source'], r['destination']) == pair), None)
            if existing is None:
                sid = ids[node]
                existing = dict(index=len(edges), source=pair[0], destination=pair[1],
                                source_station=-1 if inlet else sid, destination_station=sid if inlet else -1,
                                kind='inlet' if inlet else 'exit', road=edge['road'], direction=edge['direction'],
                                path_index=edge['path_index'], weight=0. if inlet else 1., ramp_source_ids=[],
                                boundary_basis='known_ramp_lumped_to_adjacent_observed_node_assumption')
                edges.append(existing)
            existing['ramp_source_ids'].append(int(ramp['station_id']))
            boundary[node] = True
        outgoing = Counter(r['source'] for r in edges if r['source'] >= 0)
        for edge in edges:
            if edge['source'] >= 0:
                edge['weight'] = 1./outgoing[edge['source']]
    relevant_ramps = []
    for ramp in raw:
        if ramp['Type'] not in ('On Ramp', 'Off Ramp'):
            continue
        group = groups.get((ramp['Fwy Name'], ramp['Direction']), [])
        if group and min(float(public[s]['Abs PM']) for s in group) <= float(ramp['Abs PM']) <= max(float(public[s]['Abs PM']) for s in group):
            relevant_ramps.append(int(ramp['station_id']))
    attached = {sid for edge in edges for sid in edge['ramp_source_ids']}
    return dict(station_ids=ids, edges=edges, operator_mask=mask, boundary_nodes=boundary,
                candidate_pairs=pairs, all_metadata_chains=paths, admitted_chains=admitted,
                duplicate_numeric_source_groups=list(aliases),
                ramp_policy=ramp_policy,
                relevant_ramp_source_ids=relevant_ramps,
                unmapped_ramp_source_ids=[sid for sid in relevant_ramps if sid not in attached],
                outgoing_allocation='fixed_equal_split_between_internal_and_aggregate_exit; sensitivity_required',
                evidence_scope='candidate_unverified',
                direction_rule='N/E ascending and S/W descending Abs PM; unverified hypothesis',
                physical_grid_certified=False, direct_topology_certified=False)


def point_segment_km(latitude, longitude, left, right):
    """Local geographic cross-check, not road length or driving distance."""
    lat = np.radians(latitude)
    scale = np.array([111.195, 111.195*np.cos(lat)])
    a = (np.array([float(left['Lat']), float(left['Lng'])])-[latitude, longitude])*scale
    b = (np.array([float(right['Lat']), float(right['Lng'])])-[latitude, longitude])*scale
    delta = b-a
    denominator = float(delta @ delta)
    fraction = np.clip(-float(a @ delta)/denominator, 0., 1.) if denominator > 0 else 0.
    return float(np.linalg.norm(a+fraction*delta))


def associate_reports(reports, graph, published, radius_km=1.):
    require(np.isfinite(radius_km) and radius_km > 0, 'Invalid association radius')
    public = unique_index(published)
    result = []
    for report in reports:
        item = dict(report, edge_index=-1, distance_km=0., confidence=0.)
        matches = []
        for edge in graph['edges']:
            if edge['kind'] != 'internal' or road_key(edge['road'], edge['direction']) != (int(report['road_number']), report['direction']):
                continue
            a, b = public[edge['source_station']], public[edge['destination_station']]
            pa, pb, pm = float(a['Abs PM']), float(b['Abs PM']), float(report['postmile'])
            # Half-open in travel order: a report at a shared station belongs to its outgoing edge.
            if not ((pa <= pm < pb) if pa < pb else (pb < pm <= pa)):
                continue
            distance = point_segment_km(float(report['latitude']), float(report['longitude']), a, b)
            if distance <= radius_km:
                matches.append((edge['index'], distance))
        require(len(matches) <= 1, 'Ambiguous report-to-edge association')
        if matches:
            index, distance = matches[0]
            item.update(edge_index=index, distance_km=distance, confidence=float(np.exp(-distance/radius_km)))
        result.append(item)
    return result


def cutoff_collections(events, reports, lookback_minutes=60):
    """A source report is visible iff t0-lookback <= recorded_dt <= t0.

    recorded_dt availability and immutable location are explicit assumptions.
    No eventual duration is used to declare a report active or cleared.
    """
    require(type(lookback_minutes) is int and lookback_minutes > 0, 'Invalid report lookback')
    cutoff_times = sorted({r['t0'] for r in events})
    require(all(datetime.fromisoformat(t).tzinfo is None for t in cutoff_times), 'Naive nominal cutoff calendar required')
    order = sorted(range(len(reports)), key=lambda i: (reports[i]['report_time'], reports[i]['incident_id']))
    times = [datetime.fromisoformat(reports[i]['report_time']) for i in order]
    require(all(t.tzinfo is None for t in times), 'Naive nominal report calendar required')
    offsets, indices, lo, hi = [0], [], 0, 0
    for stamp in cutoff_times:
        cutoff = datetime.fromisoformat(stamp)
        while hi < len(order) and times[hi] <= cutoff:
            hi += 1
        while lo < hi and times[lo] < cutoff-timedelta(minutes=lookback_minutes):
            lo += 1
        indices.extend(order[lo:hi])
        offsets.append(len(indices))
    mapping = {t: i for i, t in enumerate(cutoff_times)}
    counts = Counter(r['t0'] for r in events)
    return dict(cutoffs=cutoff_times, offsets=offsets, report_indices=indices,
                sample_cutoff_indices=[mapping[r['t0']] for r in events],
                unique_cutoff_sensitivity_weights=[1./counts[r['t0']] for r in events])


class CapacityNetworkInputs:
    """Consumes a hash-verified M4.1 pack and the unchanged original X adapter."""
    def __init__(self, directory, original, *, allow_exploratory=False, profile=None):
        require(allow_exploratory is True, 'M4.1 graph/report assumptions require explicit exploratory opt-in')
        self.directory, self.original = Path(directory), original
        self.summary = read_json(self.directory/'summary.json')
        require(self.summary['schema'] == SCHEMA and self.summary['main_training_ready'] is False,
                'Unexpected network pack scope')
        for name, digest in self.summary['outputs_sha256'].items():
            require(Path(name).name == name and sha256(self.directory/name) == digest, 'Network pack checksum mismatch')
        for key, digest in original.fingerprints.items():
            require(self.summary['original_inputs_sha256'][key] == digest, 'Original input identity mismatch')
        self.profile = profile if profile is not None else InformationProfile()
        require(isinstance(self.profile, InformationProfile), 'Expected InformationProfile')
        available = self.summary['information_available']
        self.information_status = {
            key: dict(available=available[key], requested=requested, enabled=requested and available[key],
                      missing_reason=None if available[key] else 'source metadata unavailable')
            for key, requested in (('new_reports', self.profile.new_reports), ('ramp_exchanges', self.profile.ramp_exchanges))}
        graph_name = 'graph.json' if self.information_status['ramp_exchanges']['enabled'] else 'conservative_graph.json'
        self.structure = read_json(self.directory/graph_name)
        require(self.structure['station_ids'] == original.station_ids.tolist(), 'Network station axis mismatch')
        self.collections = read_json(self.directory/'cutoffs.json')
        require(self.collections['sample_indices'] == [int(r['sample_index']) for r in original.events], 'Network sample axis mismatch')
        self.reports = read_rows(self.directory/'reports.csv')
        # Each graph has its own edge axis. Recompute from the same report
        # catalog; never reuse edge indices from a different information profile.
        self.reports = associate_reports(self.reports, self.structure, original.published)
        self._validate()

    def _validate(self):
        c = self.collections
        require(len(c['sample_cutoff_indices']) == len(self.original.events), 'Cutoff sample mapping mismatch')
        require(len(c['offsets']) == len(c['cutoffs'])+1 and c['offsets'][0] == 0
                and c['offsets'][-1] == len(c['report_indices'])
                and all(a <= b for a, b in zip(c['offsets'], c['offsets'][1:])), 'Invalid report offsets')
        require(len({r['incident_id'] for r in self.reports}) == len(self.reports), 'Report entities not deduplicated')
        for i, event in enumerate(self.original.events):
            k = c['sample_cutoff_indices'][i]
            require(0 <= k < len(c['cutoffs']) and c['cutoffs'][k] == event['t0'], 'Cutoff identity mismatch')
        for k, stamp in enumerate(c['cutoffs']):
            selected = c['report_indices'][c['offsets'][k]:c['offsets'][k+1]]
            require(len(selected) == len(set(selected)), 'Duplicate entity at cutoff')
            cutoff = datetime.fromisoformat(stamp)
            for index in selected:
                require(type(index) is int and 0 <= index < len(self.reports), 'Unknown report index')
                age = (cutoff-datetime.fromisoformat(self.reports[index]['report_time'])).total_seconds()/60
                require(0 <= age <= self.summary['policy']['lookback_minutes'], 'Future or expired report in cutoff set')

    def graph(self, device='cpu'):
        g = self.structure
        edges = torch.tensor([[r['source'], r['destination']] for r in g['edges']], dtype=torch.long, device=device).reshape(-1, 2).T
        graph = ExchangeGraph(len(g['station_ids']), edges,
                              torch.tensor(g['operator_mask'], dtype=torch.bool, device=device),
                              torch.tensor(g['boundary_nodes'], dtype=torch.bool, device=device),
                              evidence_scope='candidate_unverified')
        return graph, torch.tensor([r['weight'] for r in g['edges']], dtype=torch.float32, device=device)

    def batch(self, indices, device='cpu'):
        batch = self.original.batch(indices, device)
        c = self.collections
        selected = []
        for i in indices:
            k = c['sample_cutoff_indices'][int(i)]
            selected.append(c['report_indices'][c['offsets'][k]:c['offsets'][k+1]]
                            if self.information_status['new_reports']['enabled'] else [])
        b, r, e = len(indices), max(map(len, selected), default=0), len(self.structure['edges'])
        weights, distance = np.zeros((b, r, e), np.float32), np.zeros((b, r, e), np.float32)
        ages, confidence, present = np.zeros((b, r), np.float32), np.zeros((b, r), np.float32), np.zeros((b, r), bool)
        for j, group in enumerate(selected):
            cutoff = datetime.fromisoformat(self.original.events[int(indices[j])]['t0'])
            for k, index in enumerate(group):
                row = self.reports[index]
                present[j, k] = True
                ages[j, k] = (cutoff-datetime.fromisoformat(row['report_time'])).total_seconds()/60
                edge = int(row['edge_index'])
                if edge >= 0:
                    require(edge < e and self.structure['edges'][edge]['kind'] == 'internal', 'Report targets non-internal edge')
                    weights[j, k, edge] = confidence[j, k] = float(row['confidence'])
                    distance[j, k, edge] = float(row['distance_km'])
        batch['capacity_inputs']['reports'] = {key: torch.as_tensor(value, device=device) for key, value in
                                              dict(weights=weights, ages=ages, present=present, distance=distance, confidence=confidence).items()}
        batch['information_status'] = self.information_status
        return batch
