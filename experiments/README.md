# LASO Experiments — one command per paper experiment

Each script reproduces one experiment from the LASO paper by configuring `../run.py`
entirely through environment variables (see `common.sh`). Run a script from anywhere:

```bash
export LLMSQL_DATA_ROOT=/path/to/data_root     # SemBench/LRO-Bench + embedding models
# serve the expert MLLM at http://localhost:8000/v1 (see ../README.md R5), then:
bash experiments/01_main_system_comparison.sh
```

Preview the exact configuration a script will use **without** launching Spark/LLM:

```bash
LASO_DRYRUN=1 bash experiments/01_main_system_comparison.sh
```

## Experiment → paper artifact map

| Script | Paper artifact | What it varies |
|--------|----------------|----------------|
| `01_main_system_comparison.sh` | Table 2, Fig 1–2 | LASO (router/no-router) + UQE via run.py + Lotus/ThalamusDB via SemBench; shared 1%/10% per-query budget |
| `02_sampling_policy_ablation.sh` | Table 3 | sampler: uniform / top-K / UQE-active / LASO-active (LP off) |
| `03_label_propagation_ablation.sh` | Table 4 | sampler × label-propagation on/off |
| `04_online_router_ablation.sh` | Table 5 | online router off vs on |
| `05_hyperparameter_sensitivity.sh` | Fig 3 | k ∈ {10,30,50}, ρ ∈ {0.3,0.5,0.7}; per-query budget axis (mmqa_q7/ecomm_q8/ecomm_q9: 0.1–1.0%, match305/animals_q1: 1–10%) |
| `06_embedding_model_robustness.sh` | Table 6 | embedder: miniLM / CLIP-b / CLIP-l / SigLIP / GME |
| `07_mllm_generalization.sh` | Fig 4 | expert MLLM: Qwen3-VL-30B / Qwen3.6-35B / InternVL3.5-30B / GLM-4.5V |
| `08_time_breakdown.sh` | Fig 5–6 | per-stage latency decomposition (reuses 01's config) |
| `09_case_study_low_selectivity.sh` | Table 7, Fig 7 | budget B ∈ {1,5,10}% on low-selectivity queries |

Outputs land under `../results/<NN_experiment_name>/` as per-query `*_results.csv`,
`*_grid_summary.csv`, and `*_grid_summary.xlsx` (the XLSX has a `phase_breakdown` sheet
used by experiment 08).

## Generating the figures from the CSVs

After a run, build the paper figures from the per-query CSVs:

```bash
# Cost–quality Pareto frontiers
python ../pareto_frontier.py --results-dir ../results/01_main_system_comparison --output-dir ../results/pareto
# Router / delegation curves (Table 5 / router figure)
python ../delegation_curves.py --results-dir ../results/04_online_router_ablation --output-dir ../results/delegation --budgets 20,30
# Similarity-separability analysis (motivation figures)
python ../viz_similarity_distribution.py
```

## Prerequisites & scope

- **Data:** `$LLMSQL_DATA_ROOT/SemBench/files/<scenario>/{data,raw_results/ground_truth}`
  and the LRO-Bench tables (see `../README.md` R4).
- **Embedding models:** under `$LLMSQL_DATA_ROOT` (SigLIP by default; experiment 06 also
  needs CLIP-base/large, GME, miniLM).
- **Expert MLLM:** an OpenAI-compatible server at `:8000` (experiment 07 serves four in turn).
- **Baselines.** Experiment 01 now runs the full comparison: UQE rides the same `run.py`
  budget machinery as LASO, and Lotus/ThalamusDB run through the vendored SemBench harness
  (`third_party/sembench`, needs the `sembench` env — point at it with `SEMBENCH_PY`; covers
  the 10 SemBench queries only, the 4 LRO joins are unsupported there). BARGAIN still runs
  from its own repo (`third_party/bargain`) with the matching `--budget-pcts`; see
  `../README.md` "Reproducing the Baselines".

## How configuration works

`run.py` reads these variables (each defaults to its current hardcoded value, so an unset
environment reproduces the original behavior). The scripts set only what they vary:

| Variable | Controls |
|----------|----------|
| `LASO_QUERIES` | query set (ids or group names) |
| `LASO_ABLATIONS` | comma list of ablation labels `sampler[+cs][+trust_proxy|+learned_deferral|+lr_only]` |
| `LASO_BUDGET_PCTS` | global budget sweep |
| `LASO_QUERY_BUDGET_JSON` | per-query budget overrides (JSON) |
| `LASO_KNN_K` | k-NN graph k values |
| `LASO_VS_RATIO` | DIRECT explore rate ρ |
| `LASO_EMBEDDING_MODEL` / `LASO_EMB_MULTIMODAL` / `LASO_EMB_TEXT` | embedding model(s) |
| `SEMBENCH_MODEL` | expert MLLM id |
| `LASO_RESULTS_DIR` | output directory |
| `LASO_DRYRUN=1` | print resolved config and exit (no Spark/LLM) |
