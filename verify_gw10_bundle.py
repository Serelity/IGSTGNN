"""Server source-byte check. Does not import models, NumPy or test modules."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'GW10_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'gw10_residual_probe_source_manifest_v1':
        raise ValueError('Unexpected GW10 manifest schema')
    failures = []
    for name,entry in manifest['files'].items():
        path = (root/name).resolve()
        if root not in path.parents or not path.is_file():
            failures.append(dict(file=name,reason='missing_or_outside_repository')); continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            failures.append(dict(file=name,reason='byte_fingerprint_differs'))
    print(json.dumps(dict(status='GW10_SOURCE_BYTES_PASS' if not failures else 'GW10_SOURCE_BYTES_FAILED',
        checked_files=len(manifest['files']),failures=failures,model_execution=False)))
    return int(bool(failures))


if __name__ == '__main__': sys.exit(main())
