#!/usr/bin/env bash
# Experiment 01 — Main system comparison.
# Paper artifact : Table 2 (tab:sys-summary), Figure (fig:sys-f1), Figure (fig:sys-latency)
# Queries        : the 14 workload queries (SemBench 10 + LRO-Bench 4)
#
# Runs every system in the comparison:
#   * LASO + router   (activeSampling+refine+learned_deferral)  — via run.py
#   * LASO - router   (activeSampling+refine+trust_proxy)       — via run.py
#   * UQE             (uqe_active+lr_only)                       — via run.py
#   * Lotus           — via the vendored SemBench harness (third_party/sembench)
#   * ThalamusDB      — via the vendored SemBench harness (third_party/sembench)
#
# Fair-budget protocol
# --------------------
# The budget-controlled systems (UQE, BARGAIN, LASO) use the SAME per-query
# sampling budget: 1% on the six high-cardinality queries with the largest
# candidate sets (mmqa_q7, ecomm_q7, ecomm_q8, ecomm_q9, cars_q4, cars_q8) and
# 10% on the remaining eight queries. UQE and LASO read this budget from the
# SAME env vars below (LASO_BUDGET_PCTS + LASO_QUERY_BUDGET_JSON), so their
# budgets are identical by construction. (Run BARGAIN with the matching
# --budget-pcts from third_party/bargain; see README "Reproducing the Baselines".)
#
# Lotus and ThalamusDB are NOT budget-controlled — they run their own full
# evaluation — and the SemBench harness covers only the 10 SemBench queries; the
# four LRO-Bench join queries are unsupported there (shown as ✗ in the paper).
#
# Flags
#   RUN_LASO_UQE=0           skip the run.py systems (LASO±router + UQE)
#   RUN_SEMBENCH_BASELINES=0 skip Lotus + ThalamusDB
#   SEMBENCH_PY=<python>     interpreter with the `sembench` conda env (default: python3)
#   SEMBENCH_MODEL=<id>      expert MLLM (default: Qwen/Qwen3-VL-30B-A3B-Instruct)
#
# Prereqs : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output  : results/01_main_system_comparison/ (LASO+UQE); third_party/sembench (Lotus/ThalamusDB)
source "$(dirname "$0")/common.sh"

# ---------------------------------------------------------------------------
# Budget-controlled systems via run.py: LASO(+router), LASO(-router), UQE.
# All three share one invocation and therefore the identical per-query budget.
# ---------------------------------------------------------------------------
export LASO_ABLATIONS="activeSampling+refine+learned_deferral,activeSampling+refine+trust_proxy,uqe_active+lr_only"
export LASO_QUERIES="mmqa_q2a mmqa_q7 ecomm_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3 cars_q4 cars_q8 animals_q1 match113 match121 match202 match305"
export LASO_BUDGET_PCTS="10"
export LASO_QUERY_BUDGET_JSON='{"mmqa_q7":[1],"ecomm_q7":[1],"ecomm_q8":[1],"ecomm_q9":[1],"cars_q4":[1],"cars_q8":[1],"mmqa_q2a":[10],"ecomm_q13":[10],"cars_q3":[10],"animals_q1":[10],"match113":[10],"match121":[10],"match202":[10],"match305":[10]}'
export LASO_RESULTS_DIR="${RESULTS_ROOT}/01_main_system_comparison"

if [ "${RUN_LASO_UQE:-1}" = "1" ]; then
  echo "== run.py systems: LASO(+router), LASO(-router), UQE — shared per-query budget =="
  laso_run
fi

# ---------------------------------------------------------------------------
# Lotus + ThalamusDB via the vendored SemBench harness.
# Per-query scale factor matches run.py's data paths so every system reads the
# SAME table.  Columns:  paper_id  use_case  query  scale_factor
# ---------------------------------------------------------------------------
SEMBENCH_WORKLOAD=(
  "mmqa_q2a   mmqa     2a   800"
  "mmqa_q7    mmqa     7    200"
  "ecomm_q7   ecomm    7    250"
  "ecomm_q8   ecomm    8    250"
  "ecomm_q9   ecomm    9    250"
  "ecomm_q13  ecomm    13   250"
  "cars_q3    cars     3    9836"
  "cars_q4    cars     4    9836"
  "cars_q8    cars     8    9836"
  "animals_q1 animals  1    100"
)
SEMBENCH_PY="${SEMBENCH_PY:-python3}"
SEMBENCH_MODEL="${SEMBENCH_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
SEMBENCH_DIR="${REPO_ROOT}/third_party/sembench"

# Read the SemBench data in place (no re-download). The vendored harness honors
# LASO_DATA_PATH as its "files" directory; default to the standard layout under
# the data root. SEMBENCH_SKIP_SETUP=1 (default) passes --skip-setup so existing
# data is reused; set it to 0 to let SemBench download/build data on a fresh box.
export LASO_DATA_PATH="${LASO_DATA_PATH:-${LLMSQL_DATA_ROOT}/SemBench/files}"
_SKIP_SETUP=""; [ "${SEMBENCH_SKIP_SETUP:-1}" = "1" ] && _SKIP_SETUP="--skip-setup"

if [ "${RUN_SEMBENCH_BASELINES:-1}" = "1" ]; then
  echo "== SemBench baselines: Lotus + ThalamusDB (LASO_DATA_PATH=$LASO_DATA_PATH ${_SKIP_SETUP:+[reuse existing data]}) =="
  if [ "${LASO_DRYRUN:-0}" = "1" ]; then
    for row in "${SEMBENCH_WORKLOAD[@]}"; do
      read -r pid uc q sf <<< "$row"
      echo "[dryrun] lotus/thalamusdb  $pid  --use-cases $uc --queries $q --scale-factor $sf $_SKIP_SETUP"
    done
  else
    for row in "${SEMBENCH_WORKLOAD[@]}"; do
      read -r pid uc q sf <<< "$row"
      echo "-- Lotus      $pid (use-case=$uc query=$q sf=$sf)"
      ( cd "$SEMBENCH_DIR" && "$SEMBENCH_PY" src/run.py --systems lotus \
          --use-cases "$uc" --queries "$q" \
          --model "$SEMBENCH_MODEL" --scale-factor "$sf" $_SKIP_SETUP ) \
          || echo "[warn] Lotus $pid failed; continuing"
      echo "-- ThalamusDB $pid (use-case=$uc query=$q sf=$sf)"
      ( cd "$SEMBENCH_DIR" && "$SEMBENCH_PY" src/run.py --systems thalamusdb \
          --use-cases "$uc" --queries "$q" \
          --model "hosted_vllm/$SEMBENCH_MODEL" --scale-factor "$sf" $_SKIP_SETUP ) \
          || echo "[warn] ThalamusDB $pid failed; continuing"
    done
  fi
fi

echo "Experiment 01 complete. LASO/UQE → $LASO_RESULTS_DIR ; Lotus/ThalamusDB → third_party/sembench results."
