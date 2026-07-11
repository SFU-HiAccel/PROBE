#!/usr/bin/env bash
# Experiment 09 — Low-selectivity case study (limitation analysis).
# Paper artifact : Table 7 (tab:case-lowsel), Figure 7 (fig:lowsel-multimodal)
# Varies         : sampling budget B in {1%, 5%, 10%} vs the no-proxy baseline (B=100%)
# Queries        : three lower-selectivity predicates — cars_q4, ecomm_q6, movie_q5
# Prereqs        : data + embedding model under $LLMSQL_DATA_ROOT; expert MLLM @ :8000
# Output         : results/09_case_study_low_selectivity/
source "$(dirname "$0")/common.sh"

export LASO_ABLATIONS="activeSampling+refine+trust_proxy"
export LASO_QUERIES="cars_q4 ecomm_q6 movie_q5"
# Budget sweep B in {1,5,10}% for all three case-study queries. Pinned per query
# because run.py ships a built-in grid for cars_q4 ([5,10,15,20,25]) that would
# otherwise override the global sweep.
export LASO_BUDGET_PCTS="1,5,10"
export LASO_QUERY_BUDGET_JSON='{"cars_q4":[1,5,10],"ecomm_q6":[1,5,10],"movie_q5":[1,5,10]}'
# Also emit the no-proxy (full-MLLM, B=100%) baseline row for each query.
export LASO_INCLUDE_NO_OPT="1"
export LASO_RESULTS_DIR="${RESULTS_ROOT}/09_case_study_low_selectivity"

laso_run
