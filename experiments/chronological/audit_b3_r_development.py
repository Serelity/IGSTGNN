"""Train/validation-only nominal time and event audit; missing evidence stays missing."""
from datetime import datetime, timedelta


SCHEMA = 'b3r_development_time_event_audit_v1'


def audit_development(train, val, context_manifest):
    def stamp(row, key):
        return datetime.fromisoformat(row[key])

    failures = []
    for split, dataset in (('train', train), ('val', val)):
        for row in dataset.rows:
            start = stamp(row, 'x_start')
            expected = dict(x_end=start+timedelta(minutes=55), t0=start+timedelta(minutes=65),
                            y_start=start+timedelta(minutes=70), y_end=start+timedelta(minutes=125))
            if any(stamp(row, key) != value for key, value in expected.items()):
                failures.append(dict(split=split, sample_index=row['sample_index'], reason='nominal_window_axes'))
            age = (stamp(row, 't0')-stamp(row, 'report_time')).total_seconds()/60
            if not 0 < age <= 5:
                failures.append(dict(split=split, sample_index=row['sample_index'], reason='report_age'))

    def slots(dataset):
        return {stamp(row, 'y_start')+timedelta(minutes=5*i) for row in dataset.rows for i in range(12)}

    overlap = slots(train) & slots(val)
    train_end = max(stamp(row, 'y_end') for row in train.rows)
    val_cutoff = min(stamp(row, 't0') for row in val.rows)
    event_sets = [{row['incident_id'] for row in dataset.rows} for dataset in (train, val)]
    common_events = sorted(event_sets[0] & event_sets[1])
    conflict = bool(failures or overlap or train_end >= val_cutoff)
    checks = dict(
        nominal_windows_and_age=dict(status='conflict' if failures else 'confirmed', failures=failures,
                                     evidence='development manifests; reader also validates context clocks/age'),
        report_time_semantics=dict(status='missing', evidence='report_time field does not certify first publication time'),
        cutoff_report_version=dict(status='assumed', certified_initial_report=bool(context_manifest.get('source_snapshot_is_initial_report_certified', False)),
                                   evidence=context_manifest.get('assumptions', []),
                                   missing='versioned positions and publication/ingestion timestamps'),
        training_label_nominal_order=dict(status='confirmed' if train_end < val_cutoff else 'conflict',
                                         max_train_target_time=train_end.isoformat(), min_validation_cutoff=val_cutoff.isoformat()),
        label_publication_latency=dict(status='missing', evidence='no certified latency/availability timestamps'),
        shared_target_slots=dict(status='conflict' if overlap else 'confirmed', shared_nominal_time_slots=len(overlap),
                                 shared_potential_station_slot_records=len(overlap)*len(train.station_ids),
                                 examples=[v.isoformat() for v in sorted(overlap)[:20]],
                                 evidence='nominal target timestamps and common station axis; not observation publication times'),
        cross_split_event_ids=dict(status='confirmed', shared_ids=common_events, shared_count=len(common_events),
                                   train_unique_ids=len(event_sets[0]), validation_unique_ids=len(event_sets[1]),
                                   evidence='package incident_id equality; identity source/online availability not certified'),
        event_identity_provenance=dict(status='missing', evidence='package ID equality alone does not certify unseen-event generalization'),
        timezone_dst=dict(status='assumed', evidence='publisher nominal calendar; no certified timezone/DST'))
    return dict(schema=SCHEMA, status='DEVELOPMENT_CONTRACT_CONFLICT' if conflict else 'OFFLINE_AUDIT_RECORDED_WITH_MISSING_EVIDENCE',
                blocking_contract_conflict=conflict, scope='conditional_offline_screening', main_training_ready=False,
                online_time_audit_pass=False, unseen_event_generalization_certified=False,
                train_rows=len(train.rows), validation_rows=len(val.rows), stations=len(train.station_ids),
                checks=checks, test_read=False)
