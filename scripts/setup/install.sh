#!/bin/bash
# Install DUET environment.
#
# 1. Creates a venv at <repo>/venv (override with --venv <path>).
# 2. Installs torch + torchvision from the cu129 PyTorch wheel index
#    (the CUDA build vLLM 0.16.0 is compiled against).
# 3. Installs the rest of the requirements from PyPI.
# 4. Installs the prebuilt flash-attn 2.8.3 wheel for torch 2.9 (x86_64 or
#    aarch64); falls back to a source build with --no-build-isolation.
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
CUDA_INDEX="${CUDA_INDEX:-cu129}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
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
python -m pip install 'setuptools>=77.0.3,<81'

echo "[install] Installing torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION (cu_index=$CUDA_INDEX)"
pip install --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

echo "[install] Installing remaining requirements from PyPI"
# extra index so vLLM's own torch pin resolves to the CUDA build installed above
pip install --extra-index-url "https://download.pytorch.org/whl/$CUDA_INDEX" -r "$REQ_FILE"

# vLLM 0.16 uses FlashInfer (attention on Blackwell, top-p/top-k sampling), which
# JIT-compiles kernels with nvcc on first use. Compute nodes often have no CUDA
# toolkit ("Could not find nvcc"), so install FlashInfer's prebuilt kernels,
# matched to the flashinfer-python version vLLM pulled in.
FI_VER="$(python -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))')"
echo "[install] Installing prebuilt FlashInfer kernels for flashinfer $FI_VER ($CUDA_INDEX)"
pip install "flashinfer-cubin==$FI_VER" "flashinfer-jit-cache==$FI_VER" \
    --extra-index-url "https://flashinfer.ai/whl/$CUDA_INDEX"

if [[ "$SKIP_FLASH" != "1" ]]; then
    # flash-attn is used by the FSDP actor (vLLM bundles its own copy). Prefer
    # the upstream prebuilt wheel for torch 2.9 / CUDA 12 / cp312.
    FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_$(uname -m).whl"
    echo "[install] Installing flash-attn 2.8.3 ($(uname -m) wheel)"
    pip install "$FA_WHL" || {
        echo "[install] prebuilt wheel failed; building flash-attn from source (no build isolation)"
        pip install "flash-attn==2.8.3" --no-build-isolation || {
            echo "[install] flash-attn build failed; rerun with SKIP_FLASH=1 to bypass." >&2
            exit 3
        }
    }
    # flash-attn 2.8.3 also ships flash_attn/cute (FA4, Blackwell), written for an
    # older nvidia-cutlass-dsl than vLLM 0.16 installs. vLLM probes it at import
    # and only catches ImportError, so the mismatch (AttributeError: cutlass.cute.core
    # has no ThrMma) kills vLLM import. verl only uses the FA2 kernels; removing
    # the subpackage gives vLLM the same state as when flash-attn is absent.
    FA_CUTE="$(python -c 'import flash_attn, os; print(os.path.join(os.path.dirname(flash_attn.__file__), "cute"))')"
    if [[ -d "$FA_CUTE" ]]; then
        echo "[install] Removing $FA_CUTE (FA4 CuTe kernels incompatible with vLLM's cutlass-dsl)"
        rm -rf "$FA_CUTE"
    fi
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
