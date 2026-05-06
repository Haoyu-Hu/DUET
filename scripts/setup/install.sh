#!/bin/bash
# Install DUET environment.
#
# 1. Creates a venv at <repo>/venv (override with --venv <path>).
# 2. Installs torch + torchvision from the cu128 PyTorch wheel index.
# 3. Installs the rest of the requirements from PyPI.
# 4. Installs flash-attn with --no-build-isolation (must see torch).
# 5. Runs an import smoke check for torch / vllm / verl.
#
# Usage:
#   bash scripts/setup/install.sh
#   bash scripts/setup/install.sh --venv /path/to/venv
#   CUDA_INDEX=cu121 bash scripts/setup/install.sh   # alt wheel channel
#   SKIP_FLASH=1 bash scripts/setup/install.sh       # skip flash-attn build

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PATH="$REPO_ROOT/venv"
PY_BIN="${PYTHON_BIN:-python3.12}"
CUDA_INDEX="${CUDA_INDEX:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
SKIP_FLASH="${SKIP_FLASH:-0}"
REQ_FILE="$REPO_ROOT/scripts/setup/requirements.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)   VENV_PATH="$2"; shift 2 ;;
        --python) PY_BIN="$2";    shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "[install] repo_root=$REPO_ROOT"
echo "[install] venv_path=$VENV_PATH"
echo "[install] python_bin=$PY_BIN"
echo "[install] cuda_index=$CUDA_INDEX"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    echo "Error: $PY_BIN not found. Install Python 3.12 or pass --python <path>." >&2
    exit 2
fi

if [[ ! -d "$VENV_PATH" ]]; then
    echo "[install] Creating venv at $VENV_PATH"
    "$PY_BIN" -m venv "$VENV_PATH"
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Pin setuptools to satisfy both vllm (>=77.0.3) and the vendored verl (<81).
python -m pip install --upgrade pip wheel
python -m pip install 'setuptools>=77.0.3,<80'

echo "[install] Installing torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION (cu_index=$CUDA_INDEX)"
pip install --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

echo "[install] Installing remaining requirements from PyPI"
pip install -r "$REQ_FILE"

if [[ "$SKIP_FLASH" != "1" ]]; then
    echo "[install] Installing flash-attn (no build isolation)"
    pip install flash-attn==2.7.4 --no-build-isolation || {
        echo "[install] flash-attn build failed; rerun with SKIP_FLASH=1 to bypass." >&2
        exit 3
    }
fi

echo "[install] Import smoke check"
python - <<'PY'
import torch, vllm, peft
print(f"torch    {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"vllm     {vllm.__version__}")
print(f"peft     {peft.__version__}")
PY

# Verify vendored verl is importable
PYTHONPATH="$REPO_ROOT/src/verl_runtime:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import importlib
verl = importlib.import_module("verl")
print(f"verl     {getattr(verl, '__version__', '?')}  loc={verl.__file__}")
duet = importlib.import_module("duet")
print(f"duet     model_default={duet.DEFAULT_MODEL_ALIAS}")
PY

echo "[install] Done. Activate the venv with: source scripts/setup/activate.sh"
