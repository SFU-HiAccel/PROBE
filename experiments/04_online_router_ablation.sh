#!/usr/bin/env bash
# Experiment 04 — Online model-router ablation.
# Paper artifact : Table 5 (tab:router-delta)
# Varies         : router OFF (+trust_proxy) vs ON (+learned_deferral) at fixed budget
# Queries        : the 14 workload queries
# Prereqs        : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output         : results/04_online_router_ablation/
source "$(dirname "$0")/common.sh"

export LASO_ABLATIONS="activeSampling+refine+trust_proxy,activeSampling+refine+learned_deferral"
export LASO_QUERIES="mmqa_q2a mmqa_q7 ecomm_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3 cars_q4 cars_q8 animals_q1 match113 match121 match202 match305"
export LASO_BUDGET_PCTS="10"
export LASO_QUERY_BUDGET_JSON='{"ecomm_q7":[1],"mmqa_q7":[1],"cars_q4":[1],"ecomm_q8":[1],"cars_q8":[1],"ecomm_q9":[1]}'
export LASO_RESULTS_DIR="${RESULTS_ROOT}/04_online_router_ablation"

laso_run
