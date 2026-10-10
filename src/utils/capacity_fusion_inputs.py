"""Original frozen TRAIN-X adapter. Never opens flow/Y, val or test arrays."""
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from experiments.chronological.prepare_incident_corridors import load_inputs, verified
from src.models.incident_capacity_exchange import ExchangeGraph
from src.utils.incident_corridor import read_json, require, unique_index


class OriginalCapacityInputs:
    def __init__(self, data_dir, history_dir, sensors, metadata_dir, identity):
        self.data_dir = Path(data_dir)
        loaded = load_inputs(self.data_dir, Path(history_dir), Path(sensors), Path(metadata_dir), read_json(identity))
        (self.values, self.station_ids, self.events, self.trigger, self.labels,
         self.published, self.raw, self.semantics, self.fingerprints, self.source) = loaded
        try:
            self.frozen = read_json(self.data_dir / 'summary.json')
            self.fingerprints['data/scaler.json'] = verified(self.data_dir / 'scaler.json', self.frozen['files']['scaler.json'])
            self.scaler = read_json(self.data_dir / 'scaler.json')
            require(self.scaler['station_ids'] == self.station_ids.tolist()
                    and self.scaler['fitted_sample_indices'] == [int(r['sample_index']) for r in self.events]
                    and self.scaler['fit_scope'] == 'train_X_0:12_unique_station_nominal_slot_finite_nonnegative',
                    'Original flow normalization provenance mismatch')
            require(np.isfinite(self.scaler['mean']) and np.isfinite(self.scaler['std'])
                    and self.scaler['std'] > 0
                    and np.asarray(self.scaler['node_fill_mean']).shape == self.station_ids.shape
                    and np.isfinite(self.scaler['node_fill_mean']).all(), 'Invalid original flow normalization values')
            multi = read_json(Path(history_dir) / 'train_multichannel_scaler.json')
            reference = np.asarray(multi['mean'], dtype=np.float32)
            require(reference.shape == (3,) and np.isfinite(reference).all() and (reference > 0).all(),
                    'Invalid frozen training channel means')
            # Frozen original-train means, not candidate p95s and not capacity.
            self.references = np.broadcast_to(reference, (len(self.station_ids), 3)).copy()
            context = read_json(self.data_dir / 'context_manifest.json')
            self.fingerprints['data/adjacency.npy'] = verified(self.data_dir / 'adjacency.npy', context['outputs']['adjacency.npy'])
        except Exception:
            self.close()
            raise

    def batch(self, indices, device='cpu'):
        indices = np.asarray(indices)
        require(indices.ndim == 1 and len(indices) > 0 and np.issubdtype(indices.dtype, np.integer)
                and indices.min() >= 0 and indices.max() < len(self.events), 'Invalid original-train indices')
        history = np.asarray(self.values[indices]).copy()
        valid = np.isfinite(history) & (history >= 0)
        x = np.empty_like(history)
        flow = history[..., 0]
        x[..., 0] = (np.where(valid[..., 0], flow, np.asarray(self.scaler['node_fill_mean']))
                     -self.scaler['mean'])/self.scaler['std']
        for j, index in enumerate(indices):
            for k in range(12):
                dt = datetime.fromisoformat(self.events[index]['x_start']) + timedelta(minutes=5*k)
                x[j, k, :, 1] = (dt.hour*12+dt.minute//5)/288
                x[j, k, :, 2] = ((dt.weekday()+1) % 7)/7
        tensor = lambda a, dtype: torch.as_tensor(np.asarray(a).copy(), dtype=dtype, device=device)
        auxiliary = dict(history=tensor(history, torch.float32), valid=tensor(valid, torch.bool),
                         references=tensor(self.references, torch.float32),
                         labels=torch.arange(-65., -5., 5., device=device))
        trigger = {key: tensor(self.trigger[key][indices], torch.float32 if key in ('distances', 'report_age_minutes') else torch.long)
                   for key in ('distances', 'report_age_minutes', 'forecast_tod', 'forecast_dow')}
        return dict(x=tensor(x, torch.float32), incident=trigger, capacity_inputs=auxiliary)

    def qualification_ledger(self):
        """No available artifact yet certifies directed exchange/report versions."""
        from experiments.chronological.prepare_incident_physics_evidence import candidate_inventory
        public = unique_index(self.published)
        groups = Counter((public[int(s)]['Fwy'], public[int(s)]['Direction']) for s in self.station_ids)
        ordered = [public[int(s)] for s in self.station_ids]
        support = (self.trigger['distances'] != 0).any(-1)
        # Reuse metadata rules only; placeholder traffic must not be reported
        # as a measurement of joint observation coverage.
        pairs = candidate_inventory(ordered, self.raw, np.ones((1, len(ordered))), support)
        for pair in pairs:
            pair.pop('both_valid_unique_train_x_fraction')
            descending = pair['direction'] in ('S', 'W')
            pair['candidate_source_station'] = pair['high_postmile_station_id'] if descending else pair['low_postmile_station_id']
            pair['candidate_destination_station'] = pair['low_postmile_station_id'] if descending else pair['high_postmile_station_id']
            pair['travel_order_status'] = 'postmile_direction_assumption_not_direct_edge_certification'
            pair['admitted_exchange_edge'] = False
        stations = []
        for i, sid in enumerate(self.station_ids):
            row = public[int(sid)]
            stations.append(dict(station_id=int(sid), node_index=i, road=row['Fwy'], direction=row['Direction'],
                                 original_train_X_present=True,
                                 trigger_supported_train_windows=int((self.trigger['distances'][:, i] != 0).any(-1).sum()),
                                 directed_exchange_qualified=False, cutoff_visible_report_set_qualified=False,
                                 local_CTM_qualified=False,
                                 reason='direct_edges_boundary_and_source_time_evidence_pending'))
        roads = [dict(road=r, direction=d, stations=n,
                      candidate_pairs=sum(p['road'] == r and p['direction'] == d for p in pairs),
                      metadata_prefilter_pairs=sum(p['road'] == r and p['direction'] == d and p['metadata_prefilter_pass'] for p in pairs),
                      directed_exchange_qualified_stations=0,
                      status='PENDING_EVIDENCE') for (r, d), n in sorted(groups.items())]
        counts = Counter(row['t0'] for row in self.events)
        return dict(stations=stations, roads=roads, candidate_pairs=pairs,
                    summary=dict(original_train_samples=len(self.events), nodes=len(self.station_ids),
                                 road_direction_groups=len(groups), distinct_cutoffs=len(counts),
                                 extra_trigger_rows_sharing_cutoff=sum(n-1 for n in counts.values()),
                                 evaluation_policy='preserve_original_trigger_conditioned_rows; unique_cutoff_weighting_not_yet_frozen',
                                 trigger_reports_are_not_complete_visible_report_sets=True,
                                 graph_qualification_established=False,
                                 certified_directed_exchange_nodes=0,
                                 candidate_pairs=len(pairs), metadata_prefilter_pairs=sum(p['metadata_prefilter_pass'] for p in pairs),
                                 zero_certified_means_evidence_pending_not_proven_inapplicable=True,
                                 online_semantics_certified=False, main_training_ready=False))

    def qualified_graph(self, device='cpu'):
        # Evidence-pending nodes remain on the axis, with zero added increment.
        # This graph tests honest fallback only; it is NOT a coverage solution.
        n = len(self.station_ids)
        return ExchangeGraph(n, torch.empty(2, 0, dtype=torch.long, device=device),
                             torch.zeros(n, dtype=torch.bool, device=device),
                             torch.zeros(n, dtype=torch.bool, device=device),
                             evidence_scope='candidate_unverified')

    def close(self):
        self.values._mmap.close()
