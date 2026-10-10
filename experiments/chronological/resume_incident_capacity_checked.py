"""Diagnose resume identity and explicitly record GPU-name-only migration."""
import argparse
import copy
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


def gpu_name_only_change(saved, current, differences):
    return (len(differences) == 1 and differences[0]['field'] == 'environment.gpu'
            and str(saved.get('device', '')).startswith('cuda:')
            and all(isinstance(value, str) and value.strip()
                    for value in (saved.get('environment', {}).get('gpu'), current.get('environment', {}).get('gpu')))
            and current.get('device') == saved.get('device'))


def validate_runtime_segments(segments, completed, origin):
    require(isinstance(segments, list) and segments, 'Runtime segments missing')
    next_epoch = 1
    for segment in segments:
        first, last = segment['first_epoch'], segment['last_epoch']
        require(isinstance(first, int) and isinstance(last, int) and first == next_epoch and last >= first,
                'Runtime epoch segments overlap or have gaps')
        environment = segment['environment']
        current = dict(origin, environment=environment)
        differences = identity_differences(origin, current)
        require(not differences or (segment['gpu_name_change_accepted'] is True
                                    and gpu_name_only_change(origin, current, differences)),
                'Runtime segment changes more than GPU name')
        require(segment['source'] in ('origin_identity', 'checked_resume'), 'Unknown runtime segment source')
        require(segment['source'] != 'origin_identity' or not differences, 'Origin runtime environment changed')
        next_epoch = last+1
    require(next_epoch == completed+1, 'Runtime segments do not cover completed epochs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--identity-report', type=Path, required=True)
    parser.add_argument('--identity-audit-only', action='store_true')
    parser.add_argument('--allow-gpu-name-change', action='store_true',
                        help='Allow only environment.gpu to differ; preserve origin identity and record actual hardware')
    args, trainer_args = parser.parse_known_args(argv)
    require('--resume' in trainer_args and '--output-dir' in trainer_args,
            'Checked entry requires explicit --resume and --output-dir')
    root = Path(trainer_args[trainer_args.index('--output-dir')+1])
    require((root/'identity.json').is_file(), 'Existing run identity required; cannot create a new run here')
    require(not args.identity_report.exists(), 'Identity report exists; use a fresh diagnostic path')
    saved = read_json(root/'identity.json')
    original_require, original_train_arm, original_write = training.require, training.train_arm, training.write_json
    original_save = training.save_checkpoint
    captured = False
    accepted = False
    current_identity = None
    report = None
    active_arm = None

    def check_identity(condition, message):
        nonlocal captured, accepted, current_identity, report
        if message == 'Run identity changed':
            # This check is in the unchanged main, immediately before any arm update.
            current = sys._getframe(1).f_locals['identity']
            differences = identity_differences(saved, current)
            gpu_allowed = bool(args.allow_gpu_name_change and gpu_name_only_change(saved, current, differences))
            accepted = bool(condition or gpu_allowed)
            current_identity = copy.deepcopy(current)
            status = ('RESUME_IDENTITY_MATCH' if condition else
                      'RESUME_GPU_NAME_CHANGE_ACCEPTED' if gpu_allowed else 'RESUME_IDENTITY_MISMATCH')
            report = dict(status=status,
                          matching=bool(condition), differences=differences, saved_identity=saved,
                          current_identity=current, audit_only=args.identity_audit_only,
                          allow_gpu_name_change=args.allow_gpu_name_change,
                          resume_accepted=accepted, gpu_name_change_accepted=gpu_allowed,
                          origin_identity_preserved=True, bitwise_replay_claim=False,
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
            original_require(accepted, message)
            return
        original_require(condition, message)

    def checked_train_arm(model, inputs, arm, arm_root, identity, protocol,
                          seed, epochs, batch_size, device, check, resume):
        nonlocal active_arm
        require(captured and accepted, 'Training requires an accepted full identity check')
        require(Path(arm_root).resolve() == root.resolve() and resume and arm in training.ARMS,
                'Unexpected resume root, mode, or arm')
        require(identity == dict(current_identity, arm=arm), 'Current arm identity changed')
        checkpoint = root/arm/'last_checkpoint.pt'
        require(checkpoint.is_file(), 'Existing arm checkpoint required: '+arm)
        previous = training.torch.load(checkpoint, map_location='cpu', weights_only=False)
        origin_arm_identity = dict(saved, arm=arm)
        require(previous['identity'] == origin_arm_identity, 'Saved arm identity changed: '+arm)
        completed = previous['completed_epoch']
        segments = copy.deepcopy(previous.get('runtime_segments'))
        if segments is None:
            segments = [dict(first_epoch=1, last_epoch=completed,
                             environment=copy.deepcopy(saved['environment']),
                             source='origin_identity', gpu_name_change_accepted=False)] if completed else []
        if completed:
            validate_runtime_segments(segments, completed, saved)
        segment = dict(first_epoch=completed+1, last_epoch=completed,
                       environment=copy.deepcopy(current_identity['environment']),
                       source='checked_resume', gpu_name_change_accepted=report['gpu_name_change_accepted'],
                       identity_report=str(args.identity_report.resolve()),
                       identity_report_sha256=sha256(args.identity_report),
                       resume_checkpoint_sha256=sha256(checkpoint),
                       requested_absolute_epoch=epochs)
        active_arm = dict(arm=arm, directory=(root/arm).resolve(), completed=completed,
                          prior_segments=segments, segment=segment)
        del previous
        try:
            # The existing strict per-arm check still compares to the unchanged
            # checkpoint origin. Actual execution hardware is recorded separately.
            return original_train_arm(model, inputs, arm, arm_root, origin_arm_identity, protocol,
                                      seed, epochs, batch_size, device, check, resume)
        finally:
            active_arm = None

    def runtime_payload(path, value):
        if active_arm is None or Path(path).resolve().parent != active_arm['directory']:
            return value
        require(value.get('identity') == dict(saved, arm=active_arm['arm']), 'Artifact origin identity changed')
        epoch = value.get('completed_epoch', value.get('epoch'))
        require(isinstance(epoch, int) and epoch >= active_arm['completed'], 'Invalid runtime epoch')
        segments = copy.deepcopy(active_arm['prior_segments'])
        if epoch > active_arm['completed']:
            segments.append(dict(active_arm['segment'], last_epoch=epoch))
        return dict(value, runtime_segments=segments, origin_identity_preserved=True,
                    bitwise_replay_claim=False)

    def save_checkpoint(path, value):
        original_save(path, runtime_payload(path, value))

    def write_artifact(path, value):
        require(Path(path).resolve() != (root/'identity.json').resolve(), 'Cannot rewrite origin run identity')
        # Only summaries carry an identity; metrics remain in their frozen format.
        if active_arm is not None and Path(path).name == 'summary.json':
            value = runtime_payload(path, value)
        original_write(path, value)

    def reject_mutation(*_args, **_kwargs):
        raise ValueError('Identity audit must stop before training or trainer artifact writes')

    training.require = check_identity
    if args.identity_audit_only:
        training.train_arm, training.write_json, training.save_checkpoint = reject_mutation, reject_mutation, reject_mutation
    else:
        training.train_arm, training.save_checkpoint, training.write_json = checked_train_arm, save_checkpoint, write_artifact
    try:
        try:
            training.main(trainer_args)
        except IdentityAuditComplete:
            pass
        require(captured, 'Frozen trainer did not reach its identity check')
    finally:
        training.require, training.train_arm, training.write_json = original_require, original_train_arm, original_write
        training.save_checkpoint = original_save


if __name__ == '__main__':
    main()
