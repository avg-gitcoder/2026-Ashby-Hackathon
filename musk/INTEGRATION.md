# PathGen → MedGemma + MUSK

One image is used for both image models:

```text
image ── PathGen-LLaVA ── caption ── MedGemma + existing adapter ── explanation
  └───── trained MUSK ── five predictions ────────────────────────────────┘
```

The report contains the simplified explanation, followed by five named predictions as
yes/no/unknown. Original caption, probabilities and model paths are retained in JSON.
MUSK predictions are appended deterministically, without asking a language model to rewrite
them. Disagreements between the caption and classifier remain visible for human review;
this implementation does not claim to resolve them or verify caption accuracy.

## What is ready

- Existing MedGemma adapter: `../medgem/medgemma_simplifier_adapter`.
- Existing few-shot prompt: `../medgem/fewshot_instruction.txt` (English explanations).
- MUSK full checkpoint: your server's `runs/head_slurm_JOBID/best.pt`.
- New orchestration and Slurm scripts in this folder.

PathGen-1.6M is a dataset. To caption a new image, use **PathGen-LLaVA**, not PathGen-CLIP.
PathGen-LLaVA weights and a compatible LLaVA environment are NOT included.
MedGemma base weights must already be in the HF cache of the chosen environment, matching
the existing MedGemma scripts' `local_files_only=True` behavior.

## Prepare PathGen separately

Do not install LLaVA's older dependencies into the working MUSK/MedGemma environment.
Create a separate environment, following the upstream LLaVA installation instructions:

- https://github.com/haotian-liu/LLaVA#install
- https://github.com/PathFoundation/PathGen-1.6M#usage-of-trained-pathgen-llava
- https://huggingface.co/jamessyx/PathGen-LLaVA

The model download must include its `pathgenclip-vit-large-patch14` vision tower.
The official PathGen README requires the local model config's vision encoder path to
point to that downloaded vision tower. In the local model's `config.json`, update
`mm_vision_tower` (or the corresponding vision-tower field used by that checkpoint)
to the absolute path of the downloaded `pathgenclip-vit-large-patch14` directory.
Check that directory has the actual vision weights, not just a Git LFS pointer.

For example, download all files using huggingface_hub from an environment that has it:

```bash
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("jamessyx/PathGen-LLaVA", local_dir="/projects/bfqi/jkim75/models/PathGen-LLaVA"))'
```

This is a large download; check storage quota first. It does not run model inference.
Use your own approved model access where required. Do not commit model weights to GitHub.

`caption_pathgen.py` follows LLaVA's image-token, conversation-template and image-preprocessing
API. Default template is `llava_v1`; override `--conv-mode` if your checkpoint requires another.
PathGen/LLaVA environment and checkpoint compatibility still need testing on the server.

## Submit one-image report

Run from the repository's `musk` directory. Substitute the actual Python executables,
training job ID, and local PathGen model directory. Keep the absolute Python path of the
MedGemma environment where the existing `try_medgemma_fewshot.py` works.

```bash
git pull origin main
mkdir -p logs
sbatch report_delta.sbatch \
  --image /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM/TUM-YHTSLLSL.tif \
  --musk-checkpoint runs/head_slurm_TRAIN_JOB_ID/best.pt \
  --pathgen-model /projects/bfqi/jkim75/models/PathGen-LLaVA \
  --pathgen-python /absolute/path/to/pathgen-env/bin/python \
  --medgemma-python /absolute/path/to/medgemma-env/bin/python \
  --output-dir reports/example_001
```

MUSK defaults to the same module environment as the prior jobs. Override `MUSK_PYTHON`
if a different installed environment is needed. No package installation occurs inside jobs.
Each model runs in a separate process and exits before the next model loads, so the
three models do not occupy GPU memory simultaneously. Memory requirements are unverified.
Avoid submitting another GPU job for the team while this one is running.

```bash
squeue -u "$USER"
tail -f logs/report-REPORT_JOB_ID.out
cat logs/report-REPORT_JOB_ID.err
cat reports/example_001/report.txt
```

Output directory must be new. If a stage fails, the job exits and retains completed
intermediate outputs, but produces no final report. Select a new directory for a retry.
The pipeline hashes the original image and verifies image identity before merging.
Each model uses its own preprocessing on the same original image file.
No cross-model training, automated conflict reconciliation or new diagnosis is performed.

Files: `caption.json`, `explanation.json`, `musk.csv`, `report.txt`, `report.json`.
Keep these in private project storage; they are not committed by these scripts.
The checkpoint's training limitations also apply to this report.

## Verification

Python syntax and deterministic formatting / label mapping were tested locally.
The full three-model GPU pipeline, weights download and environment setup have not been
executed on Delta. A successful full run requires all three environments and weights.
