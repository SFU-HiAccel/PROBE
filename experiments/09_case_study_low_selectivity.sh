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
export LASO_BUDGET_PCTS="1,5,10"
export LASO_RESULTS_DIR="${RESULTS_ROOT}/09_case_study_low_selectivity"

laso_run
