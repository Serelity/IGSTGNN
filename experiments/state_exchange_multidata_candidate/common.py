"""Extra evaluation provenance without changing existing training fingerprints."""
from pathlib import Path

from experiments.state_exchange_multidata.common import (
    sha256, require, source_hashes as training_source_hashes)

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]


def source_hashes():
    result = training_source_hashes()
    paths = [*PACKAGE.glob('*.py'), PACKAGE/'run.sh']
    require(all(path.is_file() for path in paths), 'Missing candidate-only execution dependency')
    result.update({path.relative_to(REPO).as_posix():sha256(path) for path in sorted(paths)})
    return result
