# Changelog

## [Unreleased] - 2026-09-16

### Features
- **Learned calibration decision layer (opt-in).** Ablation suffix `+learned_calibration`
  (e.g. `LASO_ABLATIONS="activeSampling+refine+learned_calibration"`) selects
  `decision_mode="learned_calibration"`: a score-stratified calibration sample A
  (`stratified_calibration.py`), an A+L weighted Platt calibrator on the raw LR logit
  and refined label with an A-only intercept correction (`weighted_calibrator.py`),
  and exact prefix-sum dual thresholds under the remaining budget
  (`budgeted_calibrated_router.py`), orchestrated by `learned_calibration.py`.
- New result columns `lc_*` (this mode only); `policy_rollback_to_llm_count` = |A| + |D|.
- **Stratified fallback (replaces uniform exploration).** When the calibration evidence is
  insufficient (e.g. A has no positives, or misses a refined-label value), the leftover
  budget is drawn in rounds of `floor(0.1B)` from the same fixed proxy-score strata as A
  (cumulative per-stratum counts kept equal, without replacement). Each round joins A with
  weights `N_j / n_j`, the calibrator is refit, and once it succeeds the dual-threshold router
  spends the rest (`lc_status = routed_after_fallback`). If A still lacks evidence when the
  budget is exhausted, PROBE keeps refusing (`insufficient_evidence`) and outputs the refined
  labels plus every LLM label. New column `lc_fallback_rounds`.
- Env knobs: `LASO_LC_SAMPLE_FRAC` (0.6), `LASO_LC_CALIB_FRAC` (0.1).

### Design Rationale
- Total-budget semantics (`|L| + |A| + |D| <= B`) instead of R1's `B + router_cap`, as
  required by design3 §2.
- A carries known inclusion probabilities, so its weighted loss and the intercept
  correction are not biased by the active sampler's boundary enrichment; L is still used
  (weight λ, out-of-fold features, source offset δ) to reduce small-sample variance.
- Algorithm modules take proxy retraining and LLM labelling as callables, so they are
  tested offline without Spark or vLLM.

### Notes & Caveats
- Existing modes are unchanged: `predict_with_score_refinement(return_logit=False)` and
  `learned_calibration_config=None` are the defaults; no new columns for other modes.
- The active sampler's 15-call floor can exceed `floor(0.6B)` at tiny budgets; the
  overrun is reported in `lc_budget_overrun`, and A/D then receive no calls.
- `label_rows_with_llm` maps persistent request failures to label 0 and does not report
  them, so failed calibration calls cannot yet be detected (design3 §2 incomplete-sample rule).
- `lc_calls_*` count LLM-labelled rows, not request attempts: HTTP retries inside
  `label_rows_with_llm` are not charged to B (design3 §2 charges every attempt). Fixing this
  needs attempt-level accounting in the shared request layer; cross-check with
  `vllm_requests_actual`.
- `cluster_proxy_pipeline` rejects `sample_size > learned_calibration_config.sample_budget`.
- Dual-threshold ordering uses the raw proxy logit (sigmoid saturates for |x| > ~37);
  `lc_tau_plus` / `lc_tau_minus` are reported as probabilities.
- One allocation, one execution; batched re-planning (design3 §7.3) is not implemented.
- Fallback rounds stop as soon as the calibrator succeeds, so A's final size depends on its
  labels. Weights `N_j / n_j` are conditional on the realised per-stratum counts; the
  label-dependent stopping rule is not corrected for (a sequential-sampling approximation).
- Without a proxy logit (single-class L) the rounds use one stratum, i.e. uniform sampling.
- A solver failure stops the rounds; the unspent budget is left unused.
- Quality estimates (`lc_planned_f1`, `lc_estimated_f1`) are model-based, not guarantees.

## [Unreleased] - 2026-07-09

### Algorithm–paper consistency
Audited the live proxy pipeline (`proxy_filter`/`proxy_join` → `cluster_proxy_pipeline`)
against the SIGMOD submission's Algorithms 1–4 and reconciled the code to the paper.

- **Router now uses R-fold cross-fitting by default (paper Alg. 4 / §5.2).** The
  default router-training path was a fixed 6:4 held-out disjoint split — exactly the
  "reserve a fixed subset of labels" strategy §5.2 argues against. The split is now
  gated behind `cluster_proxy._ROUTER_USE_HELDOUT_SPLIT` (default `False`), so the
  router always trains on out-of-fold propagated scores over the full labeled set via
  `_compute_oof_z_smooth_and_corrected` (`R = min(5, n_pos, n_neg)` stratified folds,
  per-fold proxy retrain + re-propagation). Reverses the previous "6:4 split (#1)"
  entry below, which reflected an earlier paper draft.
- **Router feature default is the paper's 3-feature vector (Alg. 4).** When a caller
  does not pass `feature_names`, `learned_deferral.DEFAULT_DEFERRAL_FEATURES` now
  yields `(z_smooth, neighbour_disag, rank_position)` = `(f_i, nd_i(f), π_i)` instead
  of a 6-feature superset. The superset is retained as `_ALLOWED_DEFERRAL_FEATURES`
  for validating explicitly-requested features. (`run.py` already passed the 3; the
  Spark operators and direct calls defaulted to 6.)
- **Homogeneity propagation no longer clips each iteration (paper Alg. 3, Phase 2).**
  `predict_with_score_refinement` set `clip01=True` on the Phase-2 diffusion; the
  paper's update `f^(t)=(1−β)f^(0)+β S f^(t−1)` has no per-iteration clip. Now
  `clip01=False`, matching the paper (Phase-1 bias-diffusion was already unclipped).
- **Threshold-search round count uses ceil (paper Alg. 2).** `LocateThreshold`
  (`_activeSampling_version_space_reduction`) computed the geometric shrink exponent
  with floor division; paper Alg. 2 sets `m = max(1, ⌈B₁/B_batch⌉)`. Now uses
  `math.ceil`.
- **Legacy non-paper path flagged.** `proxy_model.sample_active` and
  `_online_refine` (uniform-pilot + EER sampling with a DAgger-MLP deferral head) do
  NOT implement the paper; docstring notes now point readers to the paper's
  implementations in `cluster_proxy.py`.

### Design Rationale
- Cross-fitting is the paper's stated mechanism for letting each scarce labeled row
  both anchor the proxy and yield an unbiased router target; the 6:4 split forfeits
  that under high selectivity, so it is disabled rather than deleted (flip the flag to
  reproduce the split ablation).
- The descending-vs-ascending ordering axis and the global-vs-`[I,J]` θ recovery are
  provably order-equivalent to the paper and were left in code; they are reconciled in
  the paper text rather than by risky changes to the recursive sampler.

### Notes & Caveats
- These changes alter numeric outputs of the `learned_deferral` and `refine` paths;
  re-run the affected `experiments/*.sh` (tables 2/5, figs 1–2/3/5–6) before quoting
  results.
- No automated test suite exists in the repo; changes were syntax-checked and the
  feature-default / router-flag behavior smoke-tested.

## [Unreleased] - 2026-06-22

### Documentation
- **README reframed to the ACM SIGMOD ARI author checklist.** The Reproducibility
  section is now organized around Source Code (§A), Datasets (§B), Hardware & Software
  (§C), and Experimentation/Expected Results (§D), with an artifact-metadata table and
  an ARI-item→section map. Added explicit **dataset download links**: SemBench
  ([github.com/SemBench/SemBench](https://github.com/SemBench/SemBench/),
  arXiv:2511.01716, with per-scenario Google Drive file IDs) and LRO-Bench
  ([github.com/LROBench/LROBench](https://github.com/LROBench/LROBench/),
  arXiv:2603.02537). Added a claim→command map binding each paper table/figure to one
  `experiments/*.sh` script, and dataset license notes.

### Features
- **Paper-faithful learned-deferral router (paper §3.3-§3.4).** Three algorithmic
  fixes so the default `direct+cs+learned_deferral` path matches the paper:
  - **6:4 stratified disjoint router split (#1).** New
    `learned_deferral.stratified_proxy_router_split` partitions the actively-sampled
    labels once into `L_proxy` (60%, trains the base proxy and pins Correct&Smooth)
    and a held-out `L_router` (40%, trains the router), stratified by expert-model
    label with a floor of 2 router positives. The router trains on `L_router`'s
    *unbiased* propagated scores (it was excluded from pinning), so no OOF is needed.
    Falls back to the prior 5-fold OOF over the full labeled set when the split is
    infeasible for a query/budget.
  - **Stage-2 match-side allocation (#6).** `_run_direct_single_phase` annotates
    unlabeled match-side positions in increasing distance from the threshold t*, then
    spends any leftover budget on the predicted non-match side **scanned
    deterministically from the threshold outward** (nearest/hardest negatives first)
    (Algorithm 1 L62-66). This is now the only Stage-2 behavior; the legacy symmetric
    both-sides scan and its `direct_stage2_mode` / `LASO_STAGE2_MODE` knob were removed.
  - **Router π_i rank on the blended axis (#16).** `sample_direct` returns its final
    blended ordering axis σ = (1-λ_T)z(s)+λ_T z(ℓ_T); the router ranks candidates on
    σ rather than raw similarity (similarity-rank fallback retained). Controlled by the
    new `deferral_rank_axis` param (`"blended"` default | `"similarity"`), exposed in
    `run.py` as `DEFERRAL_RANK_AXIS` / the `LASO_RANK_AXIS` env var.

- **Identifier rename to paper terminology (round 4).** Two global behavior-preserving
  renames, each verified by a fixed-seed pipeline positive-mask hash that stayed
  identical (`465e69ea3ae8ba93`) across every step:
  - **`direct` → `activeSampling`** everywhere: functions (`sample_direct`, `_direct_*`),
    params (`direct_*`), constants (`_DIRECT_*`, run.py `ACTIVE_SAMPLING_*`→`ACTIVESAMPLING_*`),
    the `"direct"` sampling-method string, and the `direct+...` ablation labels — kept in
    lockstep across run.py + `experiments/*.sh` + the parser/`_KNOWN_SAMPLERS`. Data columns
    (`Director`/`left_Director`) and English words (`directly`/`directory`) were left intact
    via word-boundary/targeted matching.
  - **"Correct & Smooth" → paper terms**: the Correct step → **Bias Diffusion**, the Smooth
    step → **Homogeneity Propagation**. Renamed `predict_with_correct_and_smooth`→
    `predict_with_score_refinement`, `cs_alpha→diffusion_alpha`, `cs_beta→propagation_beta`,
    `cs_enabled→refine_enabled`, `cs_num_iter/tol/patience→refine_*`, `cs_prof→refine_prof`,
    `_cs_methods/_cs_imbalance→_refine_*`, the `+cs` ablation component → `+refine` (run.py +
    `experiments/*.sh`), profiling keys (`correct_time→bias_diffusion_time`,
    `smooth_time→homogeneity_propagation_time`, `*_iters` likewise), and all "C&S" prose.
    `z_smooth`/`z_corrected` were intentionally kept (the score + deferral-feature contract,
    not the method name).

- **GAMCAL removal + active-learning renaming (round 3).** Stripped the entire dead
  GAMCAL dual-threshold and DASP/top-k machinery and renamed the live concepts:
  - run.py: removed `GAMCAL_DECISION_MODE`/`GAMCAL_VERIFY_UNCERTAIN` (inlined literal /
    default), `SELF_JOIN_QUERIES` (inlined the set), `ABLATION_OVERRIDES` (redundant);
    renamed `DIRECT_* → ACTIVE_SAMPLING_*`; `dasp_topk` already maps to `learned_deferral`.
  - cluster_proxy / proxy_model / sem_ops / learned_deferral: renamed live switches
    `gamcal_enabled→defer_enabled`, `gamcal_decision_mode→decision_mode`,
    `gamcal_verify_uncertain→verify_uncertain`; deleted the unreachable dasp / cold-start /
    GAM decision branches (~480 lines), the DASP pre-C&S split, the dead GAM params
    (`gamcal_alpha/beta/n_knots/lam/use_stochastic/de_*`) and DASP params
    (`dasp_proxy_ratio`, `deferral_budget_pct/rule/target_recall/fbeta/threshold`), the
    `cs_calibration` import, and the dead functions (`fit_dasp_deferral_classifier`,
    `pick_defer_threshold`, `calibrated_recall_threshold`, `f_beta_threshold`,
    `top_k_defer_mask`). Renamed profiling keys `gamcal_*→defer_*` and the `DIRECT`
    branding in comments/logs to "active-learning sampling". Net: −1077/+132 lines, zero
    `gamcal` left. The internal `sample_direct`/`direct_*`/`"direct"` contract is kept
    (tied to `experiments/*.sh` + ablation labels). Verified byte-identical: the
    `learned_deferral` pipeline's positive-mask hash is unchanged across every step.

- **`run.py` config slimming (round 2).** Inlined behavior-preserving knobs and dropped
  their constants: `CLUSTER_MODES`→`"propagation"`, `FEATURE_MODES`→`"embedding"` (only
  `[0]` was ever used), `LR_CLASS_WEIGHT`→`"balanced"`, `LR_C`→`1.0`, `BASE_THRESHOLD`/
  `JOIN_SIM_MODE`→function defaults (`None`/`"pair"`), the `DEFERRAL_RANK_AXIS` env block
  →default `"blended"`, and the five router scalars (`DEFERRAL_INNER_CV_K/MIN_INNER_CV_AUC/
  MIN_PROBE_AUC/DAGGER_BETA/K_FOLDS`)→function defaults (the `deferral_dagger_beta` default
  was changed `0.10`→`0.00` in all three layers to keep the prior probe-off behavior).
  Renamed `GAMCAL_ENABLED`→`enable_defer`. Removed the now-unused `_resolve_base_threshold`.
- **Removed per-query tuning.** Deleted `QUERY_OVERRIDES` and set a uniform
  `confidence_threshold=0.5`, `cs_alpha=cs_beta=0.8` for all queries (filter and join;
  join previously defaulted to 0.9). This is the paper's untuned config and **changes
  per-query F1 numbers** versus the tuned defaults — an intentional break, approved.
- **Unified `dasp_topk`→`learned_deferral`.** `+dasp_topk` ablations now map to the
  learned-deferral router; the unreachable `dasp_deferral` decision path was dropped from
  the run.py helpers. (The DASP code in cluster_proxy/learned_deferral is now unreachable
  from run.py but left in place; it can be deleted in a dedicated dead-code pass.)

- **`run.py` config unification (round 1).** Trimmed the experiment driver to the
  variables the active path actually consumes. Removed the dead GAM dual-threshold knobs
  (`GAMCAL_ALPHA/BETA/N_KNOTS/LAM/USE_STOCHASTIC/DE_POPSIZE/DE_MAXITER/ALPHAS`), the
  MLP router grids (`DEFERRAL_HIDDEN_DIMS_GRID/ALPHA_GRID` — the router is LR), the
  DASP/top-k knobs (`DASP_PROXY_RATIO`, `DEFERRAL_RULE/BUDGET_PCT/TARGET_RECALL/FBETA/
  THRESHOLD/THRESHOLDS`), the dead `DEFERRAL_NESTED_OOF`, the inert α/threshold sweep
  (now a single pass), and the unused multi-mode/interval + B1/explore/phase-split
  sampler knobs (rely on `cluster_proxy_pipeline` defaults). Kept: the 4 active sampler
  knobs (`DIRECT_VS_RATIO/N_PHASES/LR_BLEND_SCHEDULE/COLD_START_PHASES`) and the
  learned-deferral router config (`GAMCAL_ENABLED/DECISION_MODE/VERIFY_UNCERTAIN`,
  `DEFERRAL_FEATURES/RANK_AXIS/INNER_CV_K/MIN_INNER_CV_AUC/MIN_PROBE_AUC/DAGGER_BETA/
  K_FOLDS`).

### Design Rationale
- **Disjoint split over OOF for the router.** The paper's 6:4 split removes
  proxy-into-router leakage structurally (L_router never pins propagation), which is
  cleaner than the OOF workaround and lets the router train on the exact inference-time
  z^prop. The OOF path is kept only as a small-sample fallback so high-selectivity
  queries at tiny budgets still produce a router.

### Notes & Caveats
- The 6:4 split reduces the proxy's pinned label set to 60%; on very low-selectivity
  queries at small budgets the split may be infeasible and the run logs
  `router_split_succeeded=False` / `deferral_train_source="oof_5fold"`.
- New profiling keys: `router_split_succeeded`, `router_split_info`,
  `deferral_train_source`, `deferral_rank_axis`.

## [Unreleased] - 2026-06-20

### Features
- **Vendored baselines** under `third_party/`: code-only, anonymized copies of
  **SemBench** (`third_party/sembench`, Lotus + ThalamusDB) and **BARGAIN**
  (`third_party/bargain`), each cloned fresh from upstream with `.git`, bulk data, and
  author/website artifacts stripped. SemBench data is git-ignored and re-downloaded on
  first run; only query definitions and `raw_results/ground_truth/` labels are kept.
- **README "Reproducing the Baselines (Qwen3-VL)"** section: shared vLLM serving stanza,
  a canonical query-ID map (paper ID ↔ SemBench ↔ BARGAIN ↔ `run.py`), and run commands
  for Lotus, ThalamusDB, BARGAIN, and UQE (`LASO_ABLATIONS=uqe_active+lr_only`).
- **README "Downloading the benchmark data"**: SemBench auto-download via its harness and
  LRO-Bench/BIRD database placement under `$LLMSQL_DATA_ROOT/databases`.

### Design Rationale
- **Vendor in-tree, scrub `.git`** (vs. live submodule): a double-blind artifact cannot
  point a submodule at the author-named upstream remote, so the cleanest anonymous form is
  a plain code-only subdirectory with no remote/history.
- **Lotus and ThalamusDB documented as separate commands**: the SemBench harness shares one
  `--model` flag, but the Lotus runner prepends `hosted_vllm/` itself while ThalamusDB keys
  its model-config on the already-prefixed name, so the two need different `--model` values.

### Notes & Caveats
- Anonymity sweep: all tracked files are free of author identifiers and personal paths;
  the R5 serving block now uses `$HOME` / `$CONDA_PREFIX` / `$LLMSQL_DATA_ROOT`.
- Internal brainstorming spec/plan/prompt under `docs/superpowers/` and
  `docs/BASELINE_DOC_PLAN_PROMPT.md` are git-ignored (kept locally, not shipped) because
  they reference upstream repo URLs and paths.
- Baseline commands are verified to import / show `--help`; full runs require a live
  Qwen3-VL vLLM endpoint and a GPU.

## [Unreleased] - 2026-06-19

### Features
- **Renamed** `run_direct_galaxy_grid.py` → `run.py` (all internal references updated:
  `run_lrobench_select127_grid.py`, `run_configured_runs_xlsx.py`, the scorer test, and
  docstrings in the plot scripts).
- **Environment-variable config contract** in `run.py`: `LASO_QUERIES`, `LASO_ABLATIONS`,
  `LASO_BUDGET_PCTS`, `LASO_QUERY_BUDGET_JSON`, `LASO_KNN_K`, `LASO_VS_RATIO`,
  `LASO_EMBEDDING_MODEL` / `LASO_EMB_MULTIMODAL` / `LASO_EMB_TEXT`, `SEMBENCH_MODEL`,
  `LASO_RESULTS_DIR`, and `LASO_DRYRUN`. Every variable defaults to the prior hardcoded
  value, so an unset environment reproduces previous behavior byte-for-byte.
- **`experiments/` directory**: one shell script per LASO paper experiment (01–09), each
  mapping to its figure/table, plus `common.sh`, `README.md`, and the planning artifacts
  (`PLAN_PROMPT.md`, `PLAN.md`).
- Registered the existing `run_ecomm_q6` runner in `QUERY_RUNNERS` (needed by the
  low-selectivity case study, experiment 09).
- Plumbed the DIRECT explore rate `DIRECT_VS_RATIO` (ρ, default 0.5) into the
  `sem_join` / `sem_filter` calls so experiment 05 can sweep it.

### Design Rationale
- Per-experiment **shell scripts that set env vars + call `run.py`** keep the harness as the
  single source of truth while making each paper experiment a one-line command. Defaults are
  unchanged-on-unset to eliminate behavioral drift.

### Notes & Caveats
- Conservative cleanup only: the lone "unused import" flagged by ruff
  (`register_semantic_operators`) is in fact a **required side-effect import** (the module
  registers `df.sem_filter`/`sem_join` on import) and was preserved as an explicit
  side-effect import, not deleted.
- External baseline systems (Lotus / ThalamusDB / UQE / BARGAIN) remain out of scope.

## [Unreleased] - 2026-06-08

### Features
- **Per-query selectivity reporting** in `run_direct_galaxy_grid.py`: each query now
  prints `gt_count / total` the first time it runs — cross-product size for `sem_join`,
  table size for `sem_filter` — computed once in the `run_experiment_join/filter` funnels
  and cached across the ablation grid.
- **Hyperparameter-ablation mode** (`HPARAM_ABLATION=1`) in `run_direct_galaxy_grid.py`:
  two independent 1-D sweeps — KNN graph `k ∈ {10, 30, 50}` and DIRECT explore rate
  `direct_vs_ratio ∈ {0.3, 0.5, 0.7, 0.9}` — sharing a single anchor config `(k=30, vs_ratio=0.5)`
  (6 runs/query, no duplicate anchor) at one budget. Runs **one query at a time**: each query's
  full K × vs_ratio grid is executed, then its consolidated `{query}_hparam.csv` is written before
  moving to the next query (results saved incrementally). The tidy file has columns
  `query, sweep, value, f1, latency_s, llm_calls` covering both sweeps (the shared anchor appears
  in each).
- Plumbed `direct_vs_ratio` from the grid runner into `sem_join` / `sem_filter`
  (it was already accepted downstream but never passed by the grid) and recorded
  `direct_vs_ratio` alongside `knn_k` in every per-experiment result row.
- **cars_q4 now scored by row-level F1.** `_score_cars_q4` was changed from the
  average-age relative-error metric to precision/recall/F1 of the predicted positive
  `car_id`s against `Q4_ground_truth_rows_sf9836.csv` (per-row ground truth), mirroring
  `_score_cars_q1`. The scorer now consumes `gt_path` (previously ignored) and prints
  `GT / Pred / TP` counts so a dataset scale-factor mismatch is immediately visible.
- **Cars benchmark switched to scale factor `sf_9836`** (was `sf_157376`) across all cars
  runners (q1/q3/q4/q8/q10) — data dir and filenames (`*_9836.csv`) — so predicted
  `car_id`s share the namespace of the `sf9836` ground truth. Note: the q1/q3/q8 ground-truth
  files must contain `sf_9836`-compatible ids; the per-query `GT / Pred / TP` prints will
  surface any remaining mismatch.

### Design Rationale
- The ablation pipeline is pinned to **clean DIRECT + C&S + trust_proxy** (GAMCAL/deferral OFF):
  `direct_vs_ratio` only affects the DIRECT sampler, and the deferral cascade would add oracle
  calls that mask the K / vs_ratio effect on F1. The `"+trust_proxy"` ablation label makes
  `_decision_mode_for_ablation` return `"none"`, so `_cascade_decision_enabled` is `False`.
- Two **1-D sweeps sharing one anchor** (rather than a full 3×4 grid) keep cost low while still
  letting each curve pass through the common `(30, 0.5)` point; the anchor is run once and
  recovered into both pivots by value (`vs_ratio==0.5` for the K curve, `knn_k==30` for the
  vs_ratio curve).
- The sweep mutates module globals (`KNN_K_VALUES`, `DIRECT_VS_RATIO`, …) per round, mirroring
  the existing GAMCAL/deferral sweep driver, so no per-query runner needed editing.

### Notes & Caveats
- Single budget per query: **10%** for standard queries, **0.5%** for mmqa (its smaller scale).
- Cost ≈ 6 × #queries runs; each = embeddings + budget%×N oracle calls + C&S.
- Normal (non-`HPARAM_ABLATION`) behaviour is unchanged: `DIRECT_VS_RATIO` defaults to `0.5`
  (the prior hardcoded value) and the new kwarg simply forwards it.
- Selectivity `gt_count` uses the first-column convention of `calculate_f1_score`; GT-read
  failures or zero totals degrade to `N/A` rather than aborting a run.
