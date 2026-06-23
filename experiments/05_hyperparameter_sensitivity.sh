#!/usr/bin/env bash
# Experiment 05 — Hyperparameter sensitivity.
# Paper artifact : Figure (fig:hparam)
# Varies         : k-NN graph k in {10,30,50} (internal sweep), explore rate rho in
#                  {0.3,0.5,0.7} (outer loop), and sampling budget (per-query sweep)
# Queries        : 5 representative queries, each with its own budget axis —
#                    mmqa_q7 / ecomm_q8 / ecomm_q9 (large candidate sets): 0.1%..1.0%
#                    match305 / animals_q1 (smaller):                       1%..10%
# Prereqs        : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output         : results/05_hyperparameter_sensitivity/rho_<rho>/
source "$(dirname "$0")/common.sh"

export LASO_QUERIES="mmqa_q7 ecomm_q8 ecomm_q9 match305 animals_q1"
export LASO_ABLATIONS="activeSampling+refine+trust_proxy"
export LASO_KNN_K="10,30,50"          # k sweep (internal to one run)
# Per-query budget sweep (the figure's x-axis). High-cardinality queries sweep
# 0.1%..1.0%; the smaller queries sweep 1%..10%. Pinned per query because run.py
# ships built-in per-query budget grids that the env JSON merges over.
export LASO_BUDGET_PCTS="10"
export LASO_QUERY_BUDGET_JSON='{"mmqa_q7":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],"ecomm_q8":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],"ecomm_q9":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],"match305":[1,2,3,4,5,6,7,8,9,10],"animals_q1":[1,2,3,4,5,6,7,8,9,10]}'

# rho (explore rate) is a single value per run, so sweep it across invocations.
for rho in 0.3 0.5 0.7; do
  LASO_VS_RATIO="${rho}" \
  LASO_RESULTS_DIR="${RESULTS_ROOT}/05_hyperparameter_sensitivity/rho_${rho}" \
    laso_run
done
