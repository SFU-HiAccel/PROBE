#!/usr/bin/env bash
# Experiment 02 — Sampling-policy ablation.
# Paper artifact : Table 3 (tab:sampling-sampler)
# Varies         : sampler under an identical LR proxy (label propagation OFF):
#                  uniform (random) / top-K / UQE-style active / LASO active (activeSampling)
# Queries        : the 14 workload queries
# Prereqs        : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output         : results/02_sampling_policy_ablation/
source "$(dirname "$0")/common.sh"

# +lr_only disables Correct & Smooth, isolating the sampler's effect on the proxy.
export LASO_ABLATIONS="random+lr_only,top_k+lr_only,uqe_active+lr_only,activeSampling+lr_only"
export LASO_QUERIES="mmqa_q2a mmqa_q7 ecomm_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3 cars_q4 cars_q8 animals_q1 match113 match121 match202 match305"
export LASO_BUDGET_PCTS="10"
# Single operating point per query (NOT a sweep): 1% on the six high-cardinality
# queries, 10% on the other eight. All queries pinned because run.py ships built-in
# per-query budget grids that the env JSON merges over.
export LASO_QUERY_BUDGET_JSON='{"mmqa_q7":[1],"ecomm_q7":[1],"ecomm_q8":[1],"ecomm_q9":[1],"cars_q4":[1],"cars_q8":[1],"mmqa_q2a":[10],"ecomm_q13":[10],"cars_q3":[10],"animals_q1":[10],"match113":[10],"match121":[10],"match202":[10],"match305":[10]}'
export LASO_RESULTS_DIR="${RESULTS_ROOT}/02_sampling_policy_ablation"

laso_run
