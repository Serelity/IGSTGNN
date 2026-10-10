"""Local regeneration of the public location-only M4.2 development excerpt."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.prepare_incident_capacity_network import extract_reports
from src.utils.capacity_training_inputs import PREFIX_SHIFT_MINUTES, earlier_events, manifest_identity
from src.utils.incident_corridor import read_rows, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--sensors', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    events = {s: read_rows(args.data_dir/f'{s}_manifest.csv') for s in ('train', 'val')}
    catalog_queries = events['train'] + events['val'] + earlier_events(events['train'])
    m = extract_reports(args.raw, catalog_queries, read_rows(args.sensors), args.output_dir)
    (args.output_dir/'train_locations.tsv').rename(args.output_dir/'locations.tsv')
    m.pop('train_manifest_identity')
    m.update(schema='incident_development_location_excerpt_m42_v1',
             manifest_identity={s: manifest_identity(rows) for s, rows in events.items()},
             test_accessed=False, prefix_shift_minutes=PREFIX_SHIFT_MINUTES)
    m['policy']['scope'] = 'original train/val cutoff lookbacks plus train t0-20 prefix lookbacks; no targets or test'
    write_json(args.output_dir/'manifest.json', m)
    print({'rows': m['rows'], 'file_sha256': m['file_sha256']})


if __name__ == '__main__':
    main()
