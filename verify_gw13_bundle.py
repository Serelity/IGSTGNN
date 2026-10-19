"""Server byte verification before any GW13 imports or numerical tests."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'GW13_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'gw13_trajectory_diagnostic_source_manifest_v1':
        raise ValueError('Unexpected GW13 manifest schema')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('A nonempty source manifest is required')
    failures = []
    for name, expected in files.items():
        path = (root/name).resolve()
        if root not in path.parents or not path.is_file():
            failures.append(dict(file=name, reason='missing_or_outside_repository'))
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            failures.append(dict(file=name, reason='byte_fingerprint_differs'))
    inherited_name = 'GW12_SOURCE_MANIFEST.json'
    inherited = json.loads((root/inherited_name).read_text(encoding='utf-8'))
    if inherited.get('schema') != 'gw12_frozen_readout_source_manifest_v1':
        failures.append(dict(file=inherited_name, reason='unexpected_inherited_schema'))
    if inherited_name not in files:
        failures.append(dict(file=inherited_name, reason='inherited_manifest_not_bound'))
    inherited_files = inherited.get('files')
    if not isinstance(inherited_files, dict) or not inherited_files:
        failures.append(dict(file=inherited_name, reason='empty_inherited_manifest'))
    else:
        for name, expected in inherited_files.items():
            if files.get(name) != expected:
                failures.append(dict(file=name, reason='inherited_fingerprint_changed_or_omitted'))
    print(json.dumps(dict(status='GW13_TRAJECTORY_SOURCE_BYTES_PASS' if not failures
        else 'GW13_TRAJECTORY_SOURCE_BYTES_FAILED', checked_files=len(files), failures=failures,
        model_execution=False)))
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
