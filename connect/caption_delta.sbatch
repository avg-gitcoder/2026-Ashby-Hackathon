#!/bin/bash
#SBATCH --job-name=pathgen-caption
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pathgen-%j.out
#SBATCH --error=logs/pathgen-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from connect}"
if [[ ! -f caption_pathgen.py ]]; then
    echo 'Submit from 2026-Ashby-Hackathon/connect.' >&2
    exit 1
fi
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
PATHGEN_PYTHON="${PATHGEN_PYTHON:-$SLURM_SUBMIT_DIR/.venv-pathgen/bin/python}"
"$PATHGEN_PYTHON" caption_pathgen.py "$@"
