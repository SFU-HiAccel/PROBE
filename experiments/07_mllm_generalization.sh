#!/usr/bin/env bash
# Experiment 07 — Expert-MLLM generalization.
# Paper artifact : Figure 4 (fig:mllm-generalization)
# Varies         : expert MLLM — Qwen3-VL-30B, Qwen3.6-35B, InternVL3.5-30B, GLM-4.5V
# Queries        : the 14 workload queries (each vs its same-model no-proxy baseline)
# Prereqs        : serve EACH model in turn at the endpoint (one at a time), then run;
#                  data + embedding model under $LLMSQL_DATA_ROOT
# Output         : results/07_mllm_generalization/<model-slug>/
#
# NOTE: this script assumes the MLLM currently served at :8000 matches $SEMBENCH_MODEL
# on each iteration. If you serve models manually, run one MODEL block at a time.
source "$(dirname "$0")/common.sh"

export LASO_ABLATIONS="activeSampling+refine+learned_deferral"
export LASO_QUERIES="mmqa_q2a mmqa_q7 ecomm_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3 cars_q4 cars_q8 animals_q1 match113 match121 match202 match305"
export LASO_BUDGET_PCTS="10"
# Single operating point per query (NOT a sweep): 1% on the six high-cardinality
# queries, 10% on the other eight. All queries pinned because run.py ships built-in
# per-query budget grids that the env JSON merges over.
export LASO_QUERY_BUDGET_JSON='{"mmqa_q7":[1],"ecomm_q7":[1],"ecomm_q8":[1],"ecomm_q9":[1],"cars_q4":[1],"cars_q8":[1],"mmqa_q2a":[10],"ecomm_q13":[10],"cars_q3":[10],"animals_q1":[10],"match113":[10],"match121":[10],"match202":[10],"match305":[10]}'

for MODEL_ID in \
  "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  "Qwen/Qwen3.6-35B-A3B" \
  "OpenGVLab/InternVL3.5-30B-A3B" \
  "zai-org/GLM-4.5V"; do
  slug="$(echo "${MODEL_ID}" | tr '/' '_')"
  echo "### Expert MLLM = ${MODEL_ID}  (ensure it is served at ${LLM_ENDPOINT})"
  SEMBENCH_MODEL="${MODEL_ID}" \
  LASO_RESULTS_DIR="${RESULTS_ROOT}/07_mllm_generalization/${slug}" \
    laso_run
done
