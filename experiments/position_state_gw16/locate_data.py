"""Find the exact six-file release by hashes; read-only, no array unpickling."""
import argparse
import os
from pathlib import Path
import sys

from .common import RELEASE, sha256


def locate(roots):
    candidates = set()
    for item in roots:
        root = Path(item).expanduser().resolve()
        if (root/'incident_all.npy').is_file():
            candidates.add(root)
        if not root.is_dir():
            continue
        for parent, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in ('.git', '__pycache__', '.conda', 'node_modules'))
            if 'incident_all.npy' in files:
                candidates.add(Path(parent).resolve())
            # A worktree can expose data through a directory symlink. Resolve
            # that explicit dataset candidate without traversing symlink loops.
            for d in directories:
                child = Path(parent)/d
                if child.is_symlink():
                    for relative in ('', 'Contra_Costa', 'xtraffic/Contra_Costa', 'data/xtraffic/Contra_Costa'):
                        candidate = child/relative
                        if (candidate/'incident_all.npy').is_file():
                            candidates.add(candidate.resolve())
    fingerprints = {}
    for directory in sorted(candidates, key=lambda p:(len(p.parts), str(p))):
        match = True
        for name, expected in RELEASE.items():
            path = directory/name
            if not path.is_file():
                match = False
                break
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in fingerprints:
                fingerprints[key] = sha256(path)
            if fingerprints[key] != expected:
                match = False
                break
        if match:
            return directory
    raise FileNotFoundError('No exact Contra_Costa release found. Supply its actual directory with --data-dir; '
                            'do not rebuild, renormalize, or substitute same-shape arrays.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', nargs='+', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(locate(args.search_root))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
