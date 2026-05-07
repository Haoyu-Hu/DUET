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
    # flash-attn imports torch in setup.py, so build isolation must be off.
    # Pin range, not exact: PyPI only ships 2.7.4.post1 (no plain 2.7.4).
    echo "[install] Installing flash-attn (>=2.7.4,<3, no build isolation)"
    pip install "flash-attn>=2.7.4,<3" --no-build-isolation || {
        echo "[install] flash-attn build failed; rerun with SKIP_FLASH=1 to bypass." >&2
        exit 3
    }
fi

# Patch vLLM 0.9.2's ovis.py to add exist_ok=True on the aimv2 registration.
# Transformers >= 4.51 registers aimv2 natively; without exist_ok=True the
# duplicate registration raises:
#   ValueError: 'aimv2' is already used by a Transformers config, pick another name.
# Fixed upstream in vLLM 0.10+; we patch in place since we pin <0.10 for verl
# compatibility. Idempotent: skipped if exist_ok=True already present.
OVIS_PY="$VENV_PATH/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py"
if [[ -f "$OVIS_PY" ]]; then
    if grep -qF 'AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)' "$OVIS_PY"; then
        echo "[install] $OVIS_PY already patched (aimv2 exist_ok=True)"
    elif grep -qF 'AutoConfig.register("aimv2", AIMv2Config)' "$OVIS_PY"; then
        echo "[install] Patching $OVIS_PY (aimv2 register exist_ok=True)"
        sed -i 's|AutoConfig\.register("aimv2", AIMv2Config)|AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)|' "$OVIS_PY"
    else
        echo "[install] WARNING: aimv2 register line not found in $OVIS_PY; vLLM layout may have changed"
    fi
else
    echo "[install] WARNING: $OVIS_PY missing; vllm not installed?"
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
