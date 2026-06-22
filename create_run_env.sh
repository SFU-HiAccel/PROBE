#!/usr/bin/env bash
# create_run_env.sh — (re)create the conda env "run", a clone of the `sembench`
# env used to produce the LASO paper results.
#
# Each run DELETES an existing "run" env first, then rebuilds it from scratch.
#
# Strategy (two layers, pip with --no-deps):
#   1. conda creates the compiled base (CUDA 12.6, compilers, faiss, numpy,
#      scipy, sklearn, matplotlib, ...) from environment.run.yml.
#   2. pip installs the EXACT frozen set from requirements.run.lock.txt with
#      --no-deps. The recorded set is internally inconsistent per package
#      metadata (lotus-ai/palimpzest/thalamusdb/... declare contradictory pins),
#      so NO strict resolver can rebuild it; --no-deps installs it verbatim.

set -euo pipefail

ENV_NAME="run"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="$SCRIPT_DIR/environment.run.yml"
LOCK="$SCRIPT_DIR/requirements.run.lock.txt"

# --- locate conda ---
if ! command -v conda >/dev/null 2>&1; then
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$base/etc/profile.d/conda.sh"; break
        fi
    done
fi
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found on PATH."; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

[ -f "$YML" ]  || { echo "ERROR: $YML not found.";  exit 1; }
[ -f "$LOCK" ] || { echo "ERROR: $LOCK not found."; exit 1; }

# --- delete any existing "run" env ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Removing existing env '$ENV_NAME' ..."
    conda env remove -n "$ENV_NAME" -y
fi

echo ">>> [1/2] Creating conda base '$ENV_NAME' from $YML ..."
conda env create -n "$ENV_NAME" -f "$YML"

echo ">>> [2/2] Installing pip layer (--no-deps) from $LOCK ..."
conda run -n "$ENV_NAME" python -m pip install --no-deps -r "$LOCK"

echo ">>> pip check (the listed conflicts are known and harmless):"
conda run -n "$ENV_NAME" python -m pip check || true

echo ""
echo ">>> Done. Verify with:"
echo "    conda activate $ENV_NAME"
echo "    python -c 'import numpy, pandas, torch, transformers; print(numpy.__version__)'"
echo "    pytest tests/"
echo ""
echo ">>> NOTE: flash-attn (in the lock) is a GPU build that may fail to compile."
echo ">>>       It is a guarded/optional import; if step 2 fails on it, comment"
echo ">>>       that line out of $LOCK and re-run."
