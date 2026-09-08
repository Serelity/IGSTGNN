#!/bin/bash
#SBATCH --job-name=igstgnn
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX:-}/bin/python" ]]; then
    echo "Run 'conda activate igstgnn' before submitting with sbatch --export=ALL." >&2
    exit 1
fi

# Slurm runs a spool copy of this script, so use the submission directory.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the IGSTGNN repository root.}"
export PATH="$CONDA_PREFIX/bin:$PATH"
exec bash experiments/IGSTGNN/run.sh "$@"
