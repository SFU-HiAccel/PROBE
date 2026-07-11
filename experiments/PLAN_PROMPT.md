# Planning Prompt — LASO Per-Experiment Command CLI

> Feed this prompt to a coding agent (with repo access) to (re)generate a concrete,
> step-by-step implementation plan at `experiments/PLAN.md`. It encodes the approved
> design so the plan is reproducible and reviewable *before* any code is touched.

---

## ROLE
You are a senior systems engineer working in the `LLMSQL_anon/` repository — the
anonymized SIGMOD artifact for **LASO**, a proxy-based approximate semantic query engine.

## OBJECTIVE
Produce an implementation plan (write it to `experiments/PLAN.md`) that, when executed,
makes **each of the 9 LASO paper experiments runnable by a single command**, renames the
benchmark harness, and removes provably-dead code — **without changing any experiment's
default numerical behavior.**

## REQUIRED READING (do this first, cite line numbers in the plan)
1. `docs/superpowers/specs/2026-06-19-laso-experiment-cli-design.md` — the approved design (authoritative).
2. `LASOPaper/sections/06_evaluation.tex` and `LASOPaper/tables/table3.tex`, `table4.tex` — the experiment definitions, figures, and tables.
3. `run_direct_galaxy_grid.py` — the harness. Focus on: the `__main__` dispatch block; `QUERY_RUNNERS`; the config globals `ABLATION_CONFIGS`, `ABLATION_OVERRIDES`, `_decision_mode_for_ablation`, `BUDGET_PCTS`, `QUERY_BUDGET_PCTS`, `KNN_K_VALUES`, `DIRECT_VS_RATIO`, `GAMCAL_*`, `DEFERRAL_*`, `EMBEDDING_MODEL`, `EMBEDDING_MODELS_*`, `MODEL`.
4. The scripts that reference the harness by name: `run_lrobench_select127_grid.py`, `run_configured_runs_xlsx.py`, `tests/test_animals_scorers.py`, `pareto_frontier.py`, `delegation_curves.py`, `viz_similarity_distribution.py`.

## HARD CONSTRAINTS
- **All generated code lives under `experiments/`.** The only edits outside it are: the
  `git mv` rename to `run.py`, the env-var parsing block added to `run.py`, and the
  mechanical reference updates in the files listed above.
- **Backwards-compatible defaults:** every new env var must default to the harness's
  current hardcoded value, so an unset environment reproduces today's behavior exactly.
- **Conservative cleanup only:** remove an import or function ONLY if pyflakes/grep proves
  it unused. Never remove a config path that any of the 9 experiments exercises.
- **No algorithm/default-hyperparameter changes.**

## WHAT THE PLAN MUST CONTAIN
1. **Phase list** with explicit ordering and a verification gate after each phase.
2. **Rename phase:** exact `git mv`, and the exact reference edits (file + line + old→new).
3. **Parameterization phase:** the precise env-var → global mapping from the spec, where in
   `run.py` to insert the parsing block, and how `LASO_ABLATIONS` label strings map to
   `(label, sampling_strategy, cs_enabled)` rows and router mode. Include a `LASO_DRYRUN=1`
   path that prints the resolved config + query set and exits before Spark/LLM startup.
4. **Cleanup phase:** the command to enumerate unused imports (`pyflakes run.py`), the
   grep commands proving each candidate removal is safe, and the keep-list of config paths.
5. **`experiments/` scaffolding:** `common.sh`, `README.md`, and the 9 scripts — for each
   script specify the exact env vars it sets (ablation list, budget, k, ρ, embedding,
   model, query set, results dir) derived from the paper experiment it reproduces, and the
   paper figure/table it maps to.
6. **Verification:** `py_compile`/`compileall`, `pyflakes`, `pytest tests/` (87 pass),
   dry-run of every script, and a final grep that no `run_direct_galaxy_grid` reference
   remains.
7. **Per-experiment table** mapping `experiments/NN_*.sh` → paper figure/table → varied knob
   → query set, plus the plot-generation follow-up command where applicable.

## OUTPUT
Write the plan to `experiments/PLAN.md` as an ordered, checkbox task list grouped by phase.
Each task must be concrete enough to execute without re-deriving design decisions. Do not
write any implementation code in this step — only the plan.
