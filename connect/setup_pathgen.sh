#!/bin/bash
# Run on the Delta login node FROM connect/. Installs packages; does not run inference.
set -euo pipefail
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
unset PYTHONPATH
export PYTHONNOUSERSITE=1
CONNECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${PATHGEN_ENV_DIR:-$CONNECT_DIR/.venv-pathgen}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    python -m venv "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip 'setuptools<81' wheel
# Compatibility constraints for upstream LLaVA's older Transformers/PyTorch stack.
# No --user or --system-site-packages: leave MUSK/MedGemma environments unchanged.
"$PY" -m pip install -r "$CONNECT_DIR/requirements-pathgen.txt"
"$PY" -m pip check
"$PY" -c 'import torch, transformers, llava; from llava.model.builder import load_pretrained_model; from transformers.convert_slow_tokenizer import import_protobuf; import_protobuf(); print("Import OK:", torch.__version__, transformers.__version__)'
"$PY" -m pip freeze > "$CONNECT_DIR/pathgen-environment.txt"
printf '\nPathGen Python: %s\n' "$PY"
printf 'Next: "%s" "%s/download_pathgen.py" --model-dir /projects/bfqi/jkim75/models/PathGen-LLaVA\n' "$PY" "$CONNECT_DIR"
