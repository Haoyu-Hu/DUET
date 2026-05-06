#!/bin/bash
# Activate the DUET venv with cuda-library priority that works on this host.
#
# torch 2.7.0+cu128 ships libcusparse.so.12 which resolves __nvJitLinkCreate_12_8
# against libnvJitLink.so.12. Some hosts have an older libnvJitLink.so.12.4
# loaded first via LD_LIBRARY_PATH and fail with:
#   ImportError: ...libcusparse.so.12: undefined symbol: __nvJitLinkCreate_12_8
# The pip-installed nvidia-nvjitlink-cu12 12.8.61 in the venv has the right
# symbols. We prepend the venv-bundled nvidia/<pkg>/lib directories to
# LD_LIBRARY_PATH so the loader finds them first.
#
# Usage (from any dir):
#   source <repo>/scripts/setup/activate.sh
#   DUET_VENV=/custom/path/venv source scripts/setup/activate.sh   # override

_DUET_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_DUET_VENV="${DUET_VENV:-${_DUET_REPO}/venv}"

if [[ ! -f "${_DUET_VENV}/bin/activate" ]]; then
    echo "[duet/activate] venv not found at ${_DUET_VENV}; run scripts/setup/install.sh first." >&2
    return 1
fi

# shellcheck disable=SC1091
source "${_DUET_VENV}/bin/activate"

unset PYTHONNOUSERSITE

_DUET_NV_DIR="${_DUET_VENV}/lib/python3.12/site-packages/nvidia"
if [[ -d "${_DUET_NV_DIR}" ]]; then
    _DUET_NV_LIBS=""
    for d in "${_DUET_NV_DIR}"/*/lib; do
        [[ -d "$d" ]] && _DUET_NV_LIBS="${d}:${_DUET_NV_LIBS}"
    done
    export LD_LIBRARY_PATH="${_DUET_NV_LIBS}${LD_LIBRARY_PATH:-}"
fi

# Make `from duet import ...` and `from verl import ...` work without -m.
export PYTHONPATH="${_DUET_REPO}/src:${_DUET_REPO}/src/verl_runtime:${PYTHONPATH:-}"

echo "[duet/activate] venv=${_DUET_VENV}"
echo "[duet/activate] python=$(which python)"
echo "[duet/activate] PYTHONPATH includes src/ and src/verl_runtime"
