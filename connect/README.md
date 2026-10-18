# PathGen → MedGemma + MUSK

## Browser UI

The [Ashby workspace](UI_GUIDE.md) wraps the latest notebook's same staged runner,
with an image preview, stage retries, original captions, MUSK outputs and TXT/JSON downloads.
Run `python -m streamlit run app.py` from this folder after installing
`requirements-ui.txt`. See the guide for private Delta access and environment settings.

For environment-specific packages and fixes for fairscale/PEFT/Pillow/protobuf,
see [INSTALL.md](../INSTALL.md). Notebook dependencies belong to the actual selected
kernel. PathGen's old dependencies belong only to its separate venv.

## Three-stage notebook with saved results

Open `Run_Pathology_Report.ipynb` in VS Code connected to Delta over SSH. Select a
remote Python kernel, change `IMAGE_FILENAME` in Stage 1, and run the three cells:

1. **PathGen**: generate or reuse the caption.
2. **MedGemma**: simplify the saved caption, independently of PathGen.
3. **MUSK**: predict five labels, then display the final combined report.

Each stage submits its own Slurm GPU job only when needed. If MedGemma fails,
fix its environment/weights and retry Stage 2 only. If MUSK fails, retry Stage 3 only.
A completed stage is saved on disk and reused when run again. After restarting the
kernel, run the cells in order to restore the session; matching results are loaded
from disk without rerunning their models. A recorded pending/running job is reattached,
not submitted twice. Run one notebook session at a time.

Default checkpoint: `../musk/runs/head_slurm_22984898/best.pt`. Input images are read
from the existing TUM dataset directory. Model paths can be changed in `Settings`.
Normal stage output is just the caption, explanation, or final report; logs remain
under `logs/notebook-JOBID.out` and `.err`. Inference happens in Slurm, never on the
login/notebook node. Interrupting a waiting cell requests cancellation of its job.

### Reuse a caption generated before this notebook update

In the first cell, optionally set:

```python
EXISTING_CAPTION = "reports/pathgen_retry1.json"
```

This imports that existing caption after checking image path, SHA-256, model path and
conversation template. Importing is an explicit choice: the old JSON cannot prove the
original model weights/environment version. Default `None` uses this notebook's own
persistent cache, or runs PathGen once if no matching entry exists.

### Cache behavior

Results are stored under `reports/staged/`. Source image content, stage script content,
upstream output, configured model paths and model file sizes/timestamps identify each
stage. MUSK cache does not depend on MedGemma text; its predictions can be reused if
only the explanation changes. The final report is regenerated from the selected outputs.
Damaged result files are not silently reused. Large model weights use metadata rather
than a full content hash, to avoid heavy disk reads on the login node. In-place Python
package updates are not detected: use `force=True` after changing an environment.

To regenerate intentionally, use `session.pathgen(force=True)`,
`session.medgemma(force=True)`, or `session.musk(force=True)` in the matching cell.
Then run subsequent cells to assemble the updated final report. After a failed job,
a retry writes a fresh attempt directory and keeps successful prior stages.
Deleting the `reports/staged` folder also removes the saved results. Do not do so while
jobs are active. A `submit.lock` left by a crashed kernel requires checking its recorded
job before clearing the lock; it prevents accidental duplicate submissions.

### Prepare MedGemma once

From the module environment on the login node:

```bash
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
# Log in with an account that accepted the MedGemma model terms if needed.
python -c 'from huggingface_hub import login; login()'
python check_medgemma.py --download \
  --model-dir /projects/bfqi/jkim75/models/medgemma-4b-it
```

This downloads the base model, not another adapter. Stage 2 checks dependencies and
local weights before GPU submission. Missing MedGemma weights do not stop Stage 1.
The local model folder is detected when retrying Stage 2 after downloading the weights.
Restart the notebook kernel after pulling code updates.

Local checks use simulated Slurm/model results and cover stage failure/retry, disk reuse
after kernel restart, checkpoint changes and image identity. Live Delta GPU execution
of the new staged wrapper has not yet been verified.

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
- MUSK full checkpoint: your server's `../musk/runs/head_slurm_JOBID/best.pt`.
- New orchestration and Slurm scripts in this connect/ folder; MUSK inference is called from ../musk/infer_morphology.py.

PathGen-1.6M is a dataset. To caption a new image, use **PathGen-LLaVA**, not PathGen-CLIP.
PathGen-LLaVA weights and a compatible LLaVA environment are NOT included.
MedGemma base weights must already be in the HF cache of the chosen environment, matching
the existing MedGemma scripts' `local_files_only=True` behavior.

## Prepare PathGen separately

From `connect/` on the Delta login node, the provided installer creates an isolated
`.venv-pathgen` using the module's Python. It does not use or replace the module's PyTorch.
It installs upstream LLaVA (whose declared requirements pin torch 2.1.2 and transformers
4.37.2) and pins older compatible PEFT/HF/NumPy versions. These package combinations must
still be verified on Delta; the installer stops if dependency or import checks fail.

```bash
git pull --ff-only origin main
quota
bash setup_pathgen.sh
.venv-pathgen/bin/python download_pathgen.py \
  --model-dir /projects/bfqi/jkim75/models/PathGen-LLaVA
```

Downloads are large. Run the download only after installation completes successfully.
The downloader sets the local vision-tower config, keeps `config.original.json`, and
records the exact downloaded HF revision. Re-running it uses the HF cache/local metadata.
No model is loaded for inference on the login node.

Then test ONE caption using Slurm before running the full pipeline:

```bash
mkdir -p logs reports
sbatch caption_delta.sbatch \
  --image /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM/TUM-YHTSLLSL.tif \
  --model /projects/bfqi/jkim75/models/PathGen-LLaVA \
  --output reports/pathgen_first.json
```

Inspect `logs/pathgen-JOBID.err` if it fails; on success inspect
`cat reports/pathgen_first.json`. Use a new output filename for subsequent tests.
For the integrated job, pass
`--pathgen-python "$(pwd)/.venv-pathgen/bin/python"` from `connect/`.

The installation script writes `pathgen-environment.txt` (including the installed LLaVA
Git revision). No remote installation or GPU test has been performed by this chat.

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

Run from the repository's `connect` directory. Substitute the actual Python executables,
training job ID, and local PathGen model directory. Keep the absolute Python path of the
MedGemma environment where the existing `try_medgemma_fewshot.py` works.

```bash
git pull origin main
# If currently in musk: cd ../connect
mkdir -p logs
sbatch report_delta.sbatch \
  --image /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM/TUM-YHTSLLSL.tif \
  --musk-checkpoint ../musk/runs/head_slurm_TRAIN_JOB_ID/best.pt \
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
