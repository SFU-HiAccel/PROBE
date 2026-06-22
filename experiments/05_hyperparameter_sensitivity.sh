#!/usr/bin/env bash
# Experiment 05 — Hyperparameter sensitivity.
# Paper artifact : Figure 3 (fig:hparam)
# Varies         : k-NN graph k in {10,30,50} (internal sweep), explore rate rho in
#                  {0.3,0.5,0.7} (outer loop), and sampling budget (internal sweep)
# Queries        : 5 representative queries
# Prereqs        : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output         : results/05_hyperparameter_sensitivity/rho_<rho>/
source "$(dirname "$0")/common.sh"

export LASO_QUERIES="mmqa_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3"
export LASO_ABLATIONS="activeSampling+refine+trust_proxy"
export LASO_KNN_K="10,30,50"          # k sweep (internal to one run)
export LASO_BUDGET_PCTS="5,10,15,20,25"  # budget sweep (internal to one run)

# rho (explore rate) is a single value per run, so sweep it across invocations.
for rho in 0.3 0.5 0.7; do
  LASO_VS_RATIO="${rho}" \
  LASO_RESULTS_DIR="${RESULTS_ROOT}/05_hyperparameter_sensitivity/rho_${rho}" \
    laso_run
done
