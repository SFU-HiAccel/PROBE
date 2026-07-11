#!/usr/bin/env bash
# create_deploy_env.sh — (re)create the conda env "deploy", a clone of the
# `vllm` env used to serve the LLM (vLLM / OpenAI-compatible API) for LASO.
#
# Each run DELETES an existing "deploy" env first, then rebuilds it from scratch.
#
# Strategy (two layers, pip with --no-deps):
#   1. conda creates the compiled base (CUDA, compilers, ...) from
#      environment.deploy.yml.
#   2. pip installs the EXACT frozen set from requirements.deploy.lock.txt with
#      --no-deps, so the recorded versions go in verbatim with no resolver
#      conflicts (vLLM stacks routinely have inconsistent declared pins).

set -euo pipefail

ENV_NAME="deploy"
SOURCE_ENV="vllm"          # the live env whose vLLM version "deploy" must match
VLLM_VERSION="${VLLM_VERSION:-0.20.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="$SCRIPT_DIR/environment.deploy.yml"
LOCK="$SCRIPT_DIR/requirements.deploy.lock.txt"

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

# --- delete any existing "deploy" env ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Removing existing env '$ENV_NAME' ..."
    conda env remove -n "$ENV_NAME" -y
fi

# --- determine the vLLM version to install ---
# Explicit VLLM_VERSION wins; otherwise fall back to the live "vllm" env, then lock.
VLLM_VER="$VLLM_VERSION"
if [ -n "$VLLM_VER" ]; then
    echo ">>> Using requested vLLM==$VLLM_VER (VLLM_VERSION override)."
else
    if conda env list | awk '{print $1}' | grep -qx "$SOURCE_ENV"; then
        VLLM_VER="$(conda run -n "$SOURCE_ENV" python -c \
            'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || true)"
    fi
    if [ -z "$VLLM_VER" ]; then
        VLLM_VER="$(grep -iE '^vllm==' "$LOCK" | head -n1 | cut -d= -f3)"
        echo ">>> Source env '$SOURCE_ENV' unavailable; using vLLM==$VLLM_VER from $LOCK."
    else
        echo ">>> Detected vLLM==$VLLM_VER from live env '$SOURCE_ENV'."
    fi
fi
[ -n "$VLLM_VER" ] || { echo "ERROR: could not determine a vLLM version."; exit 1; }

echo ">>> [1/3] Creating conda base '$ENV_NAME' from $YML ..."
conda env create -n "$ENV_NAME" -f "$YML"

echo ">>> [2/3] Installing pip layer (--no-deps) from $LOCK ..."
conda run -n "$ENV_NAME" python -m pip install --no-deps -r "$LOCK"

echo ">>> [3/3] Installing vLLM==$VLLM_VER with its dependencies via pip ..."
# Install WITH deps so the requested vLLM version pulls/aligns the libraries it
# needs (torch, transformers, etc.); this may upgrade/downgrade entries from the
# --no-deps lock layer above, which is intended for a clean target version.
conda run -n "$ENV_NAME" python -m pip install "vllm==$VLLM_VER"

echo ">>> pip check (any listed conflicts are known and harmless):"
conda run -n "$ENV_NAME" python -m pip check || true

echo ""
echo ">>> Done. Verify with:"
echo "    conda activate $ENV_NAME"
echo "    python -c 'import vllm; print(vllm.__version__)'"
