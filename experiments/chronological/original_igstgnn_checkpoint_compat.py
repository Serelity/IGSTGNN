"""Keep original training identity while loading Counter on PyTorch 2.3.

MultiStepLR stores a stdlib Counter. PyTorch 2.3's weights-only reader does
not allow that class and has no public safe_globals context. This bootstrap
allows that single class temporarily; it never disables weights_only or
changes a checkpoint, training module, scheduler, optimizer or RNG state.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from threading import RLock
import uuid

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_LOCK = RLock()
_KEY = 'collections.Counter'
_MISSING = object()


@contextmanager
def counter_weights_only_compat():
    """Yield the loader strategy, restoring legacy allowlists on every exit."""
    with _LOCK:
        reader = torch.serialization._weights_only_unpickler
        allowed = reader._get_allowed_globals()
        existing = allowed.get(_KEY, _MISSING)
        if existing is not _MISSING:
            if existing is not Counter:
                raise ValueError('Existing collections.Counter allowlist entry is not stdlib Counter')
            yield 'native'
            return
        public_context = getattr(torch.serialization, 'safe_globals', None)
        if callable(public_context):
            get_user_allowed = getattr(torch.serialization, 'get_safe_globals', None)
            if not callable(get_user_allowed):
                raise ValueError('Cannot preserve an unknown public weights-only allowlist')
            has_counter = False
            for item in get_user_allowed():
                if isinstance(item, tuple) and len(item) == 2:
                    item_class, item_path = item
                else:
                    item_class = item
                    item_path = getattr(item, '__module__', '')+'.'+getattr(item, '__qualname__', '')
                if item_path == _KEY:
                    if item_class is not Counter:
                        raise ValueError('User collections.Counter allowlist entry is not stdlib Counter')
                    has_counter = True
            if has_counter:
                # safe_globals removes its entries on exit, including ones
                # which existed before entering it. Reuse them untouched.
                yield 'public_existing_Counter'
                return
            with public_context([Counter]):
                yield 'public_safe_globals'
            return
        if not re.fullmatch(r'2\.3\.\d+(?:\+[^\s]+)?', str(torch.__version__)):
            raise ValueError('Counter compatibility without safe_globals is restricted to PyTorch 2.3')
        # v2.3 _get_allowed_globals is an lru-cached dictionary consulted by
        # both GLOBAL and REDUCE opcodes. Keep every other restriction intact.
        if reader._get_allowed_globals() is not allowed or not isinstance(allowed, dict):
            raise ValueError('Expected PyTorch 2.3 cached weights-only allowlist')
        allowed[_KEY] = Counter
        try:
            yield 'legacy_2_3_Counter_only'
        finally:
            allowed.pop(_KEY, None)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _observe(root):
    """Best-effort observations, not an atomic checkpoint certificate."""
    if root is None:
        return {}
    result = {}
    for name in ('identity.json', 'summary.json', 'last.pt'):
        path = root/name
        if path.is_file():
            try:
                result[name] = _sha256(path)
            except OSError:
                result[name] = 'unavailable_during_observation'
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('train', 'screen'):
        raise ValueError('Use checkpoint compatibility entry: train|screen followed by original arguments')
    kind, arguments = argv[0], argv[1:]
    module_name = ('train_original_igstgnn_baseline' if kind == 'train'
                   else 'evaluate_original_igstgnn_modules')
    entry = importlib.import_module('experiments.chronological.'+module_name)
    # Parse with the original parser, including its unchanged validation rules.
    args = entry.parse_args(arguments)
    root = args.run_dir if kind == 'train' else args.output_dir
    if kind == 'screen' and not any(arg == '--output-dir' or arg.startswith('--output-dir=') for arg in arguments):
        # The original parser generates a timestamp when output is omitted;
        # keep the second parse and receipt on that same fresh directory.
        arguments.extend(['--output-dir', str(root)])
    before = _observe(root)
    started = datetime.now(timezone.utc).isoformat()
    strategy, status, error_text = None, 'failed', None
    try:
        with counter_weights_only_compat() as strategy:
            print(json.dumps(dict(phase='checkpoint_loader', strategy=strategy,
                                  weights_only=True, checkpoint_conversion=False)), flush=True)
            result = entry.main(arguments)
        status = 'passed'
        return result
    except BaseException as error:
        error_text = type(error).__name__+': '+str(error)
        raise
    finally:
        if root.is_dir():
            receipt = dict(schema='original_igstgnn_Counter_loader_compatibility_v1', entry=kind,
                status=status, error=error_text, started_utc=started,
                finished_utc=datetime.now(timezone.utc).isoformat(), torch_version=str(torch.__version__),
                strategy=strategy, weights_only=True,
                added_class=_KEY if strategy in ('public_safe_globals', 'legacy_2_3_Counter_only') else None,
                checkpoint_conversion=False, original_training_source_migration=False,
                driver_sha256=_sha256(Path(__file__)),
                original_entry_sha256=_sha256(Path(entry.__file__)),
                before_observations=before, after_observations=_observe(root),
                observation_scope='non_atomic_outside_original_runner_lock_not_a_training_state_certificate')
            name = 'loader_compat_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'_'+uuid.uuid4().hex[:8]+'.json'
            try:
                with (root/name).open('x', encoding='utf-8') as stream:
                    json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write('\n')
            except OSError as error:
                print('Could not write loader compatibility receipt: '+str(error), file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
