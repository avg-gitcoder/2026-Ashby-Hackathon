#!/bin/bash
#SBATCH --job-name=musk-head
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/musk-%j.out
#SBATCH --error=logs/musk-%j.err

# Submit FROM the repository's musk directory, after mkdir -p logs.
# No runtime installs. MUSK_PYTHON may point to an already-installed env's Python.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the musk directory}"
if [[ ! -f train_morphology.py ]]; then
    echo 'Submit from 2026-Ashby-Hackathon/musk.' >&2
    exit 1
fi
if [[ -z "${MUSK_PYTHON:-}" ]]; then
    module load pytorch-conda/2.8
    unset PYTHONNOUSERSITE
    MUSK_PYTHON=python
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
nvidia-smi
"$MUSK_PYTHON" -c 'import torch, timm; from musk import modeling; assert torch.cuda.is_available(), "CUDA unavailable"; print("torch:", torch.__version__, "timm:", timm.__version__)'
"$MUSK_PYTHON" train_morphology.py \
    --csv data/tum_1000_histology_classification.csv \
    --image-root /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM \
    --allow-patch-split \
    --output "runs/head_slurm_${SLURM_JOB_ID}" \
    --epochs 20 --batch-size 4 --workers 4 \
    "$@"
