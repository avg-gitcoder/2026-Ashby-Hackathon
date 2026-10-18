#!/bin/bash
#SBATCH --job-name=pathology-stage
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/notebook-%j.out
#SBATCH --error=logs/notebook-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from connect}"
STAGE="$1"
PYTHON_EXEC="$2"
shift 2
if [[ "$STAGE" == pathgen ]]; then
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
else
    module load pytorch-conda/2.8
    unset PYTHONNOUSERSITE
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
"$PYTHON_EXEC" "$@"
