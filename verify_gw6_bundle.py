"""Verify delivered file bytes using only the Python standard library.

This is not a syntax, model, unit, GPU or accuracy check.
"""
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'GW6_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'gw6_server_source_manifest_v1':
        raise ValueError('Unrecognized GW6 source manifest')
    failures = []
    for relative,expected in manifest['files'].items():
        path = (root/relative).resolve()
        if root not in path.parents or not path.is_file():
            failures.append(dict(file=relative,reason='missing_or_outside_bundle'))
        elif sha256(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            failures.append(dict(file=relative,reason='byte_fingerprint_differs'))
    print(json.dumps(dict(status='GW6_SOURCE_BYTES_PASS' if not failures else 'GW6_SOURCE_BYTES_FAILED',
        checked_files=len(manifest['files']),failures=failures,
        model_or_syntax_or_unit_tests_executed=False),ensure_ascii=False))
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
