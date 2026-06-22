# Implementation Plan — LASO Per-Experiment Command CLI

Derived from `PLAN_PROMPT.md` + `docs/superpowers/specs/2026-06-19-laso-experiment-cli-design.md`.
Execute phases in order; do not start a phase until the previous phase's verification passes.

---

## Phase 1 — Rename harness to `run.py`
- [ ] `git mv run_direct_galaxy_grid.py run.py`
- [ ] Update module docstring usage examples (`run_direct_galaxy_grid.py` → `run.py`).
- [ ] `run_lrobench_select127_grid.py`: `import run_direct_galaxy_grid as parent` and
      `from run_direct_galaxy_grid import (...)` → `run`.
- [ ] `run_configured_runs_xlsx.py`: `import run_direct_galaxy_grid as grid` → `import run as grid`.
- [ ] `tests/test_animals_scorers.py`: `(_ROOT / "run_direct_galaxy_grid.py")` → `"run.py"`.
- [ ] `pareto_frontier.py`, `delegation_curves.py`, `viz_similarity_distribution.py`:
      update docstring/help mentions of the old filename.
- **Verify:** `grep -rn run_direct_galaxy_grid . --include=*.py` → empty; `python -m py_compile run.py`; `pytest tests/ -q` → 87 pass.

## Phase 2 — Env-var config contract in `run.py`
Insert one `_env_*` parsing block right after the config globals it overrides (after
`ABLATION_CONFIGS`/`BUDGET_PCTS`/`KNN_K_VALUES`/`DIRECT_VS_RATIO`/`EMBEDDING_MODEL`).
Each var defaults to the existing global so unset == current behavior.

- [ ] `LASO_BUDGET_PCTS` → `BUDGET_PCTS` (comma floats).
- [ ] `LASO_QUERY_BUDGET_JSON` → merge into `QUERY_BUDGET_PCTS` (JSON).
- [ ] `LASO_KNN_K` → `KNN_K_VALUES` (comma; `auto` or ints).
- [ ] `LASO_VS_RATIO` → `DIRECT_VS_RATIO` (comma floats).
- [ ] `LASO_EMBEDDING_MODEL` → `EMBEDDING_MODEL`.
- [ ] `LASO_EMBEDDING_SET` → choose `EMBEDDING_MODELS_TEXT|MULTIMODAL|all|csv` sweep.
- [ ] `LASO_HPARAM_ABLATION` → existing hparam mode flag.
- [ ] `LASO_RESULTS_DIR` → `results_dir`.
- [ ] `LASO_ABLATIONS` → rebuild `ABLATION_CONFIGS` from labels. Map each label:
      `direct|random|top_k|uqe_active` → `sampling_strategy`; `+cs` → `cs_enabled=True`,
      `+lr_only` → `False`; router suffix `+trust_proxy|+learned_deferral` honored by
      existing `_decision_mode_for_ablation`. Validate labels; abort on unknown.
- [ ] `LASO_ROUTER=on|off` → rewrite trailing router suffix on each ablation label.
- [ ] `LASO_QUERIES` → query set (overrides argv); supports group names + explicit ids.
- [ ] `LASO_DRYRUN=1` → after resolving config, print a config echo (queries, ablations,
      budgets, k, ρ, embedding, model, results dir) and `sys.exit(0)` before Spark/LLM init.
- **Verify:** `python -m py_compile run.py`; `LASO_DRYRUN=1 LASO_ABLATIONS=... python run.py <q>` prints expected config; `pytest tests/` still 87 pass.

## Phase 3 — Conservative dead-code cleanup
- [ ] `pyflakes run.py` → list unused imports; remove each only after confirming with grep.
- [ ] For each candidate dead function/branch: `grep -rn <name> . --include=*.py`; remove
      only if zero references AND not an experiment config path (keep `GAMCAL_*`,
      `DEFERRAL_*`, `uqe_active`/`top_k`/`random` branches, `_iter_emb_models`, hparam mode).
- [ ] Remove obviously-dead commented scaffolding only if it references deleted symbols.
- **Verify:** `pyflakes run.py` clean of unused-import warnings; `compileall .`; `pytest tests/` 87 pass.

## Phase 4 — `experiments/` scripts
- [ ] `experiments/common.sh`: `set -euo pipefail`; assert `LLMSQL_DATA_ROOT` set; remind
      to serve the MLLM at `$LLM_ENDPOINT` (default localhost:8000); `RESULTS_ROOT=./results`;
      helper `run_laso()` that `cd`s to repo root and calls `python run.py`.
- [ ] Nine scripts, each: header (paper artifact, prerequisites, expected output, est.
      runtime), `source common.sh`, export only varied vars, set `LASO_RESULTS_DIR=./results/<id>`, call run.

| Script | Sets | Paper artifact |
|--------|------|----------------|
| `01_main_system_comparison.sh` | `LASO_ABLATIONS=direct+cs+learned_deferral,direct+cs+trust_proxy`; size-aware `LASO_QUERY_BUDGET_JSON` (1% for mmqa_q7/mmqa_q2a/ecomm_q7/ecomm_q8/ecomm_q9/ecomm_q13, 10% rest); all 14 q | Table 2, Fig 1–2 |
| `02_sampling_policy_ablation.sh` | `LASO_ABLATIONS=random+lr_only,top_k+lr_only,uqe_active+lr_only,direct+lr_only`; all 14 q | Table 3 |
| `03_label_propagation_ablation.sh` | `LASO_ABLATIONS=random+lr_only,random+cs+trust_proxy,direct+lr_only,direct+cs+trust_proxy`; all 14 q | Table 4 |
| `04_online_router_ablation.sh` | `LASO_ABLATIONS=direct+cs+trust_proxy,direct+cs+learned_deferral`; all 14 q | Table 5 |
| `05_hyperparameter_sensitivity.sh` | `LASO_HPARAM_ABLATION=1`; `LASO_KNN_K=10,30,50`; `LASO_VS_RATIO=0.3,0.5,0.7`; `LASO_QUERIES=mmqa_q7 ecomm_q8 ecomm_q9 ecomm_q13 cars_q3` | Fig 3 |
| `06_embedding_model_robustness.sh` | `LASO_EMBEDDING_SET=all` (text→8 q, multimodal→10 q); `direct+cs+trust_proxy` | Table 6 |
| `07_mllm_generalization.sh` | loop `SEMBENCH_MODEL` ∈ {Qwen3-VL-30B, Qwen3.6-35B, InternVL3.5-30B, GLM-4.5V}; all 14 q | Fig 4 |
| `08_time_breakdown.sh` | `direct+cs+learned_deferral`; all 14 q; phase-timing columns | Fig 5–6 |
| `09_case_study_low_selectivity.sh` | `LASO_QUERIES=cars_q4 ecomm_q6 movie_q5`; `LASO_BUDGET_PCTS=1,5,10` | Table 7, Fig 7 |

- [ ] `experiments/README.md`: the table above + prerequisites (data, embedding models,
      MLLM endpoint) + plot follow-ups: `pareto_frontier.py` (Pareto), `delegation_curves.py`
      (router/Fig), `viz_similarity_distribution.py` (separability); note external baselines
      live in their own repos.
- [ ] `chmod +x experiments/*.sh`.
- **Verify:** `bash -n experiments/*.sh`; `LASO_DRYRUN=1 bash experiments/<id>.sh` echoes the
      intended config for every script.

## Phase 5 — Final verification & docs
- [ ] `grep -rn run_direct_galaxy_grid . --include=*.py` → empty.
- [ ] `compileall .` clean; `pytest tests/` 87 pass; `pyflakes run.py` clean.
- [ ] Update root `README.md` reproduce section: `run_direct_galaxy_grid.py` → `run.py`
      and point at `experiments/` for one-command-per-experiment.
- [ ] `CHANGELOG.md` entry.

## Rollback
All work is in this repository's git; `git restore`/`git checkout` reverts. The original
upstream development repository is never touched.
