# Delta inference

Run submission commands from the repository's `musk` directory on the login node.
Inference itself runs on the GPU node through Slurm. Use the same environment as training.
Create `logs` before submission. Replace TRAIN_JOB_ID with the completed training job ID.

```bash
git pull origin main
mkdir -p logs
ls runs/head_slurm_*/best.pt
```

## One image

```bash
sbatch infer_delta.sbatch \
  --checkpoint runs/head_slurm_TRAIN_JOB_ID/best.pt \
  --image /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM/TUM-YHTSLLSL.tif \
  --output predictions/one_image.csv
```

## Images listed in the validation CSV

```bash
sbatch infer_delta.sbatch \
  --checkpoint runs/head_slurm_TRAIN_JOB_ID/best.pt \
  --csv runs/head_slurm_TRAIN_JOB_ID/val.csv \
  --output predictions/validation.csv
```

The default image root is `/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM`.
Use `--image-root /another/directory` to change it. CSV only needs a `filename` column;
ground-truth columns, if present, are ignored. Validation results are not an independent test.

For a directory of new images, replace `--csv ...` with `--input-dir /path/to/images`.
Directory mode selects supported image files directly inside that folder, non-recursively.
Use `--image` with an absolute path for a single image.

## Watch and read results

```bash
squeue -u "$USER"
tail -f logs/musk-infer-INFERENCE_JOB_ID.out
cat logs/musk-infer-INFERENCE_JOB_ID.err
head -n 3 predictions/one_image.csv
```

The final CSV is created on success. Existing output files are not overwritten.
After a failed run choose a new output filename; an incomplete `.partial` file may remain.
Each feature has a prediction (-1/no, 0/unknown, +1/yes), the selected probability,
and all three probabilities. Softmax and argmax are applied separately per feature.
Probabilities have not been calibrated. Ties choose the first class in order no/unknown/yes.

The training `best.pt` includes the full MUSK encoder and task head, so inference does not
download official weights again. Only load checkpoints you trust.
An untrained head or head-only state dict is not accepted by this script.

The five outputs inherit training label limitations. Missing negative classes in training
cannot be validated by running inference on the same images.

Local checks cover Python syntax and file selection. GPU inference must be verified on Delta.
