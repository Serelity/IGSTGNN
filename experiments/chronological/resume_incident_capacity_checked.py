"""Report exact resume identity differences at the frozen trainer's own check."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_incident_capacity as training
from src.utils.incident_corridor import read_json, require, sha256, write_json


class IdentityAuditComplete(Exception):
    """Stop after the original trainer has constructed its full identity."""


def identity_differences(saved, current, prefix=''):
    if isinstance(saved, dict) and isinstance(current, dict):
        result = []
        for key in sorted(set(saved) | set(current)):
            field = prefix+'.'+key if prefix else key
            if key not in saved or key not in current:
                result.append(dict(field=field, saved=saved.get(key, '<missing>'), current=current.get(key, '<missing>')))
            else:
                result.extend(identity_differences(saved[key], current[key], field))
        return result
    return [] if saved == current else [dict(field=prefix, saved=saved, current=current)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--identity-report', type=Path, required=True)
    parser.add_argument('--identity-audit-only', action='store_true')
    args, trainer_args = parser.parse_known_args(argv)
    require('--resume' in trainer_args and '--output-dir' in trainer_args,
            'Checked entry requires explicit --resume and --output-dir')
    root = Path(trainer_args[trainer_args.index('--output-dir')+1])
    require((root/'identity.json').is_file(), 'Existing run identity required; cannot create a new run here')
    require(not args.identity_report.exists(), 'Identity report exists; use a fresh diagnostic path')
    saved = read_json(root/'identity.json')
    original_require, original_train_arm, original_write = training.require, training.train_arm, training.write_json
    captured = False

    def check_identity(condition, message):
        nonlocal captured
        if message == 'Run identity changed':
            # This check is in the unchanged main, immediately before any arm update.
            current = sys._getframe(1).f_locals['identity']
            differences = identity_differences(saved, current)
            report = dict(status='RESUME_IDENTITY_MATCH' if condition else 'RESUME_IDENTITY_MISMATCH',
                          matching=bool(condition), differences=differences, saved_identity=saved,
                          current_identity=current, audit_only=args.identity_audit_only,
                          checker_sha256=sha256(Path(__file__)), test_accessed=False)
            write_json(args.identity_report, report)
            captured = True
            print(report['status'], flush=True)
            for row in differences:
                print('  '+row['field']+': saved='+json.dumps(row['saved'])
                      +' current='+json.dumps(row['current']), flush=True)
            print('Identity report: '+str(args.identity_report), flush=True)
            if args.identity_audit_only:
                raise IdentityAuditComplete()
        original_require(condition, message)

    def reject_mutation(*_args, **_kwargs):
        raise ValueError('Identity audit must stop before training or trainer artifact writes')

    training.require = check_identity
    if args.identity_audit_only:
        training.train_arm, training.write_json = reject_mutation, reject_mutation
    try:
        try:
            training.main(trainer_args)
        except IdentityAuditComplete:
            pass
        require(captured, 'Frozen trainer did not reach its identity check')
    finally:
        training.require, training.train_arm, training.write_json = original_require, original_train_arm, original_write


if __name__ == '__main__':
    main()
