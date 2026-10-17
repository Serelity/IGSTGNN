"""Only train unfinished GW2 dynamic arms; reuse certified GW1 references.

The original GW2 execution files remain byte-for-byte unchanged so an existing
a2c5b32 run can resume its POST_LOCAL/POST_MESSAGE epoch checkpoints.
"""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_original_graph_message as base
from experiments.chronological.graph_message_reuse import (
    NEW_ARMS, load_reference, preflight, read_json, require, reused_summary, sidecar)
from experiments.chronological.original_igstgnn_runtime import configure_runtime, run_lock, sha256, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reuse-gw1-run', type=Path, required=True)
    p.add_argument('--gw1-best-archive', type=Path)
    p.add_argument('--gw1-last-archive', type=Path)
    p.add_argument('--data-dir', type=Path, default=base.default_data_dir())
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check', action='store_true')
    args = p.parse_args(argv)
    if not 1 <= args.epochs <= base.CONFIG['max_epochs']:
        p.error('epochs must be a total budget in 1..100')
    args.run_dir, args.reuse_gw1_run = args.run_dir.resolve(), args.reuse_gw1_run.resolve()
    require(args.run_dir != args.reuse_gw1_run, 'Source and destination runs must differ')
    require(args.run_dir.exists() == args.resume, 'Use --resume for an existing GW2 run; do not overwrite it')
    control = sidecar(args.run_dir)
    control.mkdir(parents=True, exist_ok=True)
    with run_lock(control):
        saved = control/'reference.json'
        prior = read_json(saved) if saved.exists() else None
        archives = dict(best=args.gw1_best_archive, last=args.gw1_last_archive)
        if prior:
            archives = {selection: value or prior['archives'][selection]['path']
                        for selection, value in archives.items()}
        reference, source_identity, old_states = load_reference(args.reuse_gw1_run, args.epochs,
            archives=archives)
        runtime = configure_runtime(args.device, args.threads, args.seed)
        preflight(source_identity, args, runtime)
        if saved.exists():
            require(read_json(saved) == reference, 'Reuse reference changed; preserve the original reference')
        else:
            write_json(saved, reference)
        write_json(control/'driver.json', dict(schema='gw2_reuse_driver_v1',
            execution_source={name: sha256(REPO/'experiments/chronological'/name) for name in (
                'train_original_graph_message_reuse.py', 'graph_message_reuse.py')},
            new_training_arms=list(NEW_ARMS), baseline_new_optimizer_updates=0, test_evaluated=False))
        command = ['--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir),
            '--epochs', str(args.epochs), '--seed', str(args.seed), '--device', args.device,
            '--threads', str(args.threads), '--only', *NEW_ARMS]
        if args.resume:
            command.append('--resume')
        if args.check:
            command.append('--check')
        # base.main preserves its strict checkpoint identity. --only prevents
        # ANY optimizer or validation evaluation of ORIGINAL/STATIC here.
        base.main(command)
        # Recheck the immutable references after the new training completes.
        current, source_identity, old_states = load_reference(args.reuse_gw1_run, args.epochs,
            target=read_json(args.run_dir/'identity.json'),
            archives=dict(best=reference['archives']['best']['path'], last=reference['archives']['last']['path']))
        require(current == reference, 'Source artifacts changed while new models trained')
        summary = reused_summary(args.run_dir, reference, source_identity, old_states, args.epochs)
        write_json(args.run_dir/'reuse_summary.json', summary)
        print(summary['status'], flush=True)
        return summary


if __name__ == '__main__':
    main()
