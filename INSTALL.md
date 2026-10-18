# Installation and dependency recovery (Delta)

These manifests cover the active MUSK morphology scripts, MedGemma inference/training,
PathGen captioning, and the three-stage notebook. They do not promise to install every
optional benchmark/tool copied from upstream MUSK. Use Python 3.11 on Delta.

## Environments must stay separate

| Environment | Requirements | Purpose |
|---|---|---|
| Delta runtime | `musk/requirements-delta.txt` + `medgem/requirements-inference.txt` | MUSK + MedGemma |
| Optional training add-on | `medgem/requirements-training.txt` | MedGemma training with datasets |
| Notebook kernel | `connect/requirements-notebook.txt` | Slurm orchestration + image display |
| Isolated PathGen venv | `connect/requirements-pathgen.txt` | Old LLaVA/PyTorch/Transformers stack |

Do not combine all requirements into one environment. Do not install the upstream
`musk/requirements.txt` over the working Delta module: that file preserves upstream's
torch 2.0.1 / torchvision 0.15.2 recipe, while the active Delta scripts use torch 2.8.
`constraints-delta.txt` prevents dependency resolution from replacing the Delta torch
version; it is not a standalone PyTorch/CUDA installation recipe.

## 1. Delta runtime: MUSK + MedGemma

Run from the repository root on the login node. Reuse a working installation instead
of reinstalling unnecessarily. For a new installation, create a small overlay venv:

```bash
module load pytorch-conda/2.8
unset PYTHONPATH
unset PYTHONNOUSERSITE
python -m venv --system-site-packages .venv-delta
source .venv-delta/bin/activate
python -m pip install -r musk/requirements-delta.txt -r medgem/requirements-inference.txt
python -m pip check
```

The overlay reuses module PyTorch/CUDA and isolates newly installed packages from the
shared module. `--system-site-packages` deliberately exposes module packages; it is not
an isolated clean environment. A module-level `pip check` conflict may be unrelated to
this project: inspect it rather than upgrading the entire scientific stack blindly.

No editable MUSK installation is required for the current scripts: Python imports its
local `musk/` package when the scripts run from their location. If using it outside
these scripts, install `python -m pip install -e ./musk --no-deps` separately.

For optional MedGemma training:

```bash
python -m pip install -r medgem/requirements-training.txt
```

Register the same environment as a notebook kernel:

```bash
python -m pip install -r connect/requirements-notebook.txt
python -m ipykernel install --user --name ashby-delta --display-name "Ashby Delta"
```

Select **Ashby Delta** in the remote VS Code notebook. `%pip` is a notebook-cell command,
not a Bash command. To repair a different selected notebook kernel, run
`%pip install -r /absolute/path/to/2026-Ashby-Hackathon/connect/requirements-notebook.txt`
inside that notebook. Terminal installation alone does not repair a different kernel.

### Point jobs at the overlay Python

The default job scripts and notebook settings point at the original module Python.
Installing an overlay does not automatically switch those defaults.
In the notebook Stage 1, configure before constructing `ReportSession`:

```python
settings = Settings(
    connect=_connect,
    medgemma_python=_connect.parent / ".venv-delta/bin/python",
)
```

The staged runner uses this Python for both MedGemma and MUSK. For standalone MUSK
batch jobs, set the following in the same terminal before `sbatch`:

```bash
export MUSK_PYTHON="$(pwd)/.venv-delta/bin/python"  # from repository root
```

For the old integrated CLI also pass `--medgemma-python` with the same absolute path.
Start jobs after loading the Delta module. GPU inference/training must run through Slurm.

## 2. PathGen (separate environment)

```bash
cd connect
bash setup_pathgen.sh
```

The installer creates `.venv-pathgen` without system site packages, installs
`requirements-pathgen.txt`, runs dependency/import checks, and saves a local
`pathgen-environment.txt`. The LLaVA Git commit is pinned, instead of following HEAD.
PathGen uses PEFT 0.7.1; MedGemma uses the newer adapter-compatible PEFT range. These
versions belong in different environments. PathGen's protobuf dependency is explicit.

## 3. Model files and access are separate from pip requirements

- **MUSK**: obtain individual Hugging Face access for `xiangjx/musk`. The trained
  `best.pt` includes the encoder and does not redownload base weights for inference.
- **PathGen**: follow `connect/README.md`; download weights and configure the local
  vision-tower path using `download_pathgen.py`.
- **MedGemma**: accept the terms for `google/medgemma-4b-it`, log in using that same
  account, then download the base model. The committed LoRA adapter is not the base model.

From `connect/`, using the installed Delta runtime Python:

```bash
python -c 'from huggingface_hub import login; login()'
python check_medgemma.py --download \
  --model-dir /projects/bfqi/jkim75/models/medgemma-4b-it
```

Then retry only notebook Stage 2. 401/403 permission errors cannot be fixed by installing
packages. `local_files_only=True` means base model files must already be available.
The original `medgem/train_*.py` and `try_*.py` use the default HF cache, unlike the
integrated pipeline's local-directory option. To use these standalone scripts, either
download to that cache (`snapshot_download("google/medgemma-4b-it")` without local_dir)
or change their `model_id` to the downloaded local path.

Training data for MedGemma is also external: scripts expect `~/pathology-simplifier/`
with `data_split/` or `data_split_preserve/`. Installing datasets does not supply it.

## 4. Import checks before GPU submission

With the Delta runtime active, from the repository root:

```bash
python -c 'from peft import PeftModel; from transformers import Gemma3ForConditionalGeneration; from PIL import Image; import fairscale; print("Delta imports OK")'
cd musk
python -c 'from musk import modeling; print("MUSK import OK")'
cd ../connect
python check_medgemma.py
```

The check does not load full model weights into a GPU. Use the existing caption/Slurm
test for actual inference. Run `python -m pip freeze` in each successful environment to
record exact resolved versions. Current manifests bound direct dependencies; they are
not a complete transitive lock or a guarantee of clean installation success.

## Problems from this session covered here

| Symptom | Fix recorded |
|---|---|
| No module named fairscale | MUSK Delta requirements |
| No module named peft | MedGemma requirements, separate from PathGen PEFT |
| No module named PIL | Pillow in runtime and notebook requirements |
| LlamaTokenizer requires protobuf | PathGen requirements + installer import check |
| Path \| None fails on old kernel | Existing Optional annotations; use Python 3.11 |
| Old torch/transformers conflicts | Separate PathGen env and Delta constraints |
| Missing MedGemma base cache | Explicit download and preflight instructions |
| 401 / 403 gated model | Model terms/access + same-account server login |
| %pip run in Bash | Notebook versus terminal instructions |

Optional upstream benchmarks can require CONCH/OpenCLIP/webdataset and their own model
access. Apex, FlashAttention, fairseq and Tutel imports in the copied encoder have guarded
fallbacks for the active MUSK configuration; they are not automatically required here.
Check the specific upstream benchmark/optional mode before using it.

## Verification limits

The manifests were checked against repository imports and upstream package metadata.
They have not been installed from scratch or run on a Delta GPU by this chat. Version
ranges are compatibility targets, not claims about the exact successful server versions.
Source references: https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/pyproject.toml
and the committed MedGemma adapter model card (PEFT 0.17.1).
