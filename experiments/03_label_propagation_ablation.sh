#!/usr/bin/env bash
# Experiment 03 — Label-propagation ablation.
# Paper artifact : Table 4 (tab:sampling-prop)
# Varies         : sampler {uniform, LASO active} x proxy {LR only, LR + label propagation}
# Queries        : the 14 workload queries
# Prereqs        : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output         : results/03_label_propagation_ablation/
source "$(dirname "$0")/common.sh"

# +lr_only = no propagation; +refine+trust_proxy = label propagation on, router off.
export LASO_ABLATIONS="random+lr_only,random+refine+trust_proxy,activeSampling+lr_only,activeSampling+refine+trust_proxy"
export LASO_QUERIES="mmqa_q2a mmqa_q7 ecomm_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3 cars_q4 cars_q8 animals_q1 match113 match121 match202 match305"
export LASO_BUDGET_PCTS="10"
# Single operating point per query (NOT a sweep): 1% on the six high-cardinality
# queries, 10% on the other eight. All queries pinned because run.py ships built-in
# per-query budget grids that the env JSON merges over.
export LASO_QUERY_BUDGET_JSON='{"mmqa_q7":[1],"ecomm_q7":[1],"ecomm_q8":[1],"ecomm_q9":[1],"cars_q4":[1],"cars_q8":[1],"mmqa_q2a":[10],"ecomm_q13":[10],"cars_q3":[10],"animals_q1":[10],"match113":[10],"match121":[10],"match202":[10],"match305":[10]}'
export LASO_RESULTS_DIR="${RESULTS_ROOT}/03_label_propagation_ablation"

laso_run
