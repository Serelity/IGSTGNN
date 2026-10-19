"""Locate a completed GW10 result and its exact bound cache, never by score."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.branch_readout_gw9.readout_auth import SOURCE_IDENTITY_SHA256


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def discover(probe_dir, cache_dir=None, gw10_dir=None):
    probe_dir = Path(probe_dir).resolve()
    cache_paths = [Path(cache_dir).resolve()/'cache.json'] if cache_dir else list(
        probe_dir.parent.glob(probe_dir.name+'_gw10_cache_*/cache.json'))
    result_paths = [Path(gw10_dir).resolve()/'analysis.json'] if gw10_dir else list(
        probe_dir.parent.glob(probe_dir.name+'_gw10_probe_*/analysis.json'))
    caches = {}
    for path in cache_paths:
        try:
            metadata = json.loads(path.read_text(encoding='utf-8'))
            if (metadata.get('schema')=='gw10_frozen_sum_residual_cache_v1'
                    and metadata.get('status')=='GW10_FROZEN_SUM_CACHE_COMPLETE'
                    and metadata.get('source_identity_sha256')==SOURCE_IDENTITY_SHA256
                    and all((path.parent/(name+'.npz')).is_file() for name in ('train','val'))):
                caches.setdefault(digest(path),[]).append(path.parent.resolve())
        except (OSError,ValueError,TypeError): continue
    pairs = []
    for path in result_paths:
        try:
            analysis = json.loads(path.read_text(encoding='utf-8'))
            if (analysis.get('schema')!='gw10_train_only_residual_direction_probe_v1'
                    or analysis.get('source_certificate',{}).get('source_identity_sha256')!=SOURCE_IDENTITY_SHA256
                    or not all((path.parent/name).is_file() for name in
                        ('model.json','screening.json','protocol.json','report.md'))): continue
            for cache in caches.get(analysis.get('cache_metadata_sha256'),[]):
                pairs.append((path.stat().st_mtime_ns,str(path.parent.resolve()),str(cache)))
        except (OSError,ValueError,TypeError): continue
    if not pairs:
        raise ValueError('No complete matching GW10 cache/result pair. Set GW11_CACHE_DIR and GW11_GW10_RUN '
            'to the existing completed source directories; preserve their absolute paths.')
    # Completion recency resolves duplicate equivalent sources; no metric is consulted.
    _,result,cache = max(pairs)
    return Path(cache),Path(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--gw10-dir',type=Path)
    args = parser.parse_args(argv)
    cache,result = discover(args.probe_dir,args.cache_dir,args.gw10_dir)
    # This deliberately only locates candidates; load_inputs performs full authentication.
    print(cache); print(result)


if __name__=='__main__': main()
