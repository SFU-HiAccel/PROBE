#!/usr/bin/env bash
# Shared setup for LASO experiment scripts. Sourced by experiments/NN_*.sh.
#
# Each NN_*.sh script exports only the LASO_* variables it varies, then calls
# `laso_run`. Unset variables fall back to run.py's built-in defaults, so the
# scripts change ONLY what their experiment varies.
set -euo pipefail

# Repo root = parent of the experiments/ directory.
_EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_EXP_DIR}/.." && pwd)"

# Data root must point at the SemBench / LRO-Bench + embedding-model layout
# (see ../README.md "Reproducibility R4"). Defaults to $HOME, matching run.py.
export LLMSQL_DATA_ROOT="${LLMSQL_DATA_ROOT:-$HOME}"

# Expert MLLM endpoint reminder (OpenAI-compatible). run.py posts to :8000.
LLM_ENDPOINT="${LLM_ENDPOINT:-http://localhost:8000/v1}"

# Root for experiment outputs; each script sets LASO_RESULTS_DIR beneath it.
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"

# Run the harness from the repo root. Extra args are passed through to run.py
# (normally none — the query set comes from LASO_QUERIES).
laso_run() {
  ( cd "${REPO_ROOT}" && python run.py "$@" )
}

if [ "${LASO_DRYRUN:-0}" = "1" ]; then
  echo "[common] REPO_ROOT=${REPO_ROOT}"
  echo "[common] LLMSQL_DATA_ROOT=${LLMSQL_DATA_ROOT}"
  echo "[common] LLM_ENDPOINT=${LLM_ENDPOINT}"
fi
