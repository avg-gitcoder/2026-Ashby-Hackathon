#!/bin/bash
#SBATCH --job-name=musk-infer
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/musk-infer-%j.out
#SBATCH --error=logs/musk-infer-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository musk directory}"
if [[ ! -f infer_morphology.py ]]; then
    echo 'Submit from 2026-Ashby-Hackathon/musk.' >&2
    exit 1
fi
if [[ -z "${MUSK_PYTHON:-}" ]]; then
    module load pytorch-conda/2.8
    unset PYTHONNOUSERSITE
    MUSK_PYTHON=python
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1
"$MUSK_PYTHON" infer_morphology.py "$@"
