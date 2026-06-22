#!/usr/bin/env bash
# Experiment 06 — Embedding-model robustness.
# Paper artifact : Table 6 (tab:embed)
# Varies         : embedding model — miniLM (text only), CLIP-base, CLIP-large,
#                  SigLIP, GME. run.py picks the text vs multimodal list per query.
# Queries        : text queries use the TEXT list; multimodal queries the MULTIMODAL list
# Prereqs        : ALL listed embedding-model snapshots under $LLMSQL_DATA_ROOT;
#                  data; expert MLLM @ :8000
# Output         : results/06_embedding_model_robustness/
source "$(dirname "$0")/common.sh"

D="${LLMSQL_DATA_ROOT}"
# Multimodal-capable embedders, run on the multimodal queries.
export LASO_EMB_MULTIMODAL="${D}/siglip-base-patch16-224,${D}/clip-vit-base-patch32,${D}/clip-vit-large-patch14-336,${D}/gme-Qwen2-VL-2B-Instruct"
# Text-only queries: SigLIP (text tower) + miniLM. Adjust to taste.
export LASO_EMB_TEXT="${D}/siglip-base-patch16-224,${D}/all-MiniLM-L6-v2"

export LASO_ABLATIONS="activeSampling+refine+trust_proxy"
export LASO_QUERIES="mmqa_q2a mmqa_q7 ecomm_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3 cars_q4 cars_q8 animals_q1"
export LASO_BUDGET_PCTS="10"
export LASO_RESULTS_DIR="${RESULTS_ROOT}/06_embedding_model_robustness"

laso_run
