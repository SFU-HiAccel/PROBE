#!/usr/bin/env bash
# Experiment 01 — Main system comparison.
# Paper artifact : Table 2 (tab:sys-summary), Figure 1 (fig:sys-f1), Figure 2 (fig:sys-latency)
# Varies         : LASO with router (+learned_deferral) vs without (+trust_proxy)
# Queries        : the 14 workload queries (SemBench 10 + LRO-Bench 4)
# Budget         : size-aware — 1% on the 6 high-cardinality queries, 10% elsewhere
# Prereqs        : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output         : results/01_main_system_comparison/
# Note           : external baselines (Lotus / ThalamusDB / UQE / BARGAIN) live in
#                  their own repos; this run produces the LASO + no-proxy rows.
source "$(dirname "$0")/common.sh"

export LASO_ABLATIONS="activeSampling+refine+learned_deferral,activeSampling+refine+trust_proxy"
export LASO_QUERIES="mmqa_q2a mmqa_q7 ecomm_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3 cars_q4 cars_q8 animals_q1 match113 match121 match202 match305"
# Single operating budget (10%), with 1% overrides for the high-cardinality
# queries. Tune via LASO_BUDGET_PCTS / LASO_QUERY_BUDGET_JSON.
export LASO_BUDGET_PCTS="10"
export LASO_QUERY_BUDGET_JSON='{"ecomm_q7":[1],"mmqa_q7":[1],"cars_q4":[1],"ecomm_q8":[1],"cars_q8":[1],"ecomm_q9":[1]}'
export LASO_RESULTS_DIR="${RESULTS_ROOT}/01_main_system_comparison"

laso_run
