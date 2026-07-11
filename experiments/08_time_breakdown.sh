#!/usr/bin/env bash
# Experiment 08 — Per-stage latency breakdown.
# Paper artifact : Figure 5 (fig:time-stack), Figure 6 (fig:time-abs)
# Varies         : nothing new — decomposes the full LASO run into per-phase timings
#                  (embedding / sampling / labeling / propagation / training / prediction
#                  / router rollback). The phase_* columns are emitted by every run.
# Queries        : the 14 workload queries
# Prereqs        : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output         : results/08_time_breakdown/  (see *_grid_summary.xlsx -> phase_breakdown)
#
# This reuses the main-comparison configuration; Figures 5-6 read the per-phase
# timing columns of the resulting summary, so a single LASO+router run suffices.
source "$(dirname "$0")/common.sh"

export LASO_ABLATIONS="activeSampling+refine+learned_deferral"
export LASO_QUERIES="mmqa_q2a mmqa_q7 ecomm_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3 cars_q4 cars_q8 animals_q1 match113 match121 match202 match305"
export LASO_BUDGET_PCTS="10"
# Single operating point per query (NOT a sweep): 1% on the six high-cardinality
# queries, 10% on the other eight. All queries pinned because run.py ships built-in
# per-query budget grids that the env JSON merges over.
export LASO_QUERY_BUDGET_JSON='{"mmqa_q7":[1],"ecomm_q7":[1],"ecomm_q8":[1],"ecomm_q9":[1],"cars_q4":[1],"cars_q8":[1],"mmqa_q2a":[10],"ecomm_q13":[10],"cars_q3":[10],"animals_q1":[10],"match113":[10],"match121":[10],"match202":[10],"match305":[10]}'
export LASO_RESULTS_DIR="${RESULTS_ROOT}/08_time_breakdown"

laso_run
