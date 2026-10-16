# Algorithm mapping

Authority: current manuscript worktree at `67f02b5`, including uncommitted edits.
File hashes are in `manuscript_snapshot.json`. The appendix is disabled in the
manuscript and does not override the main text.

| Manuscript requirement | Code | Implementation |
| --- | --- | --- |
| Fixed predicate-aware ranking and same-pass features | `ce_ranker.py`, `proxy_model._reduce_rows` | Qwen3-VL-Reranker-2B scores and last-token states, L2 normalization and PCA256 |
| Preparation budget | `paper_calibration_config`, `PreparationLedger` | `L = floor(3B/5)`, `A = B-L`, routing and S3 outside B |
| SAS pilot and gate | `fixed_ce_sampling.py` | Uniform pilot `floor(0.3 B_L)`, one-sided hypergeometric bound, strict lower-bound `> 0.05` |
| SAS batch cap | `SamplingConfig`, `sample_fixed_ce` | `max(1, floor(0.01 N))`, explicit batch300 for the trace figure |
| Stratified branch | `_stratum_quota`, `geometric_strata` | Pilot Neyman weights, minimum stratum exploration, capacity-bounded quotas |
| Multi-interval branch | `threshold_search_multi.py`, `_paper_sas` | 25% uniform exploration, scan until two positives, DP penalty1.5, cap6, shrinking windows and seed service |
| S3 | `_whole_query_llm_fallback` | Retain purchased labels and evaluate remaining candidates when training lacks either class |
| Proxy | `predict_with_score_refinement` | Class-balanced L2 logistic regression, C1 |
| Graph | `_build_knn_graph` | Unit embeddings, Gaussian weights, union of directed kNN edges, median kth-neighbor distance, no self edges |
| Refinement | `predict_with_score_refinement` | Row-normalized P, residual pinning, corrected-score pinning and propagated-label pinning, damping0.8, exactly50 iterations per phase |
| Calibration sampling | `StratifiedDesign`, `allocate_neyman` | Equal-width refined-score bins, adjacent merges until H<=m, floor `max(H, ceil(0.2m))`, Neyman remainder, SRSWOR weights `N_j/n_j` |
| Cross-fitting | `compute_oof_features` | `min(5,n_pos,n_neg)` folds, fold-specific LR and refinement, held-out labels excluded |
| Platt model and loss | `weighted_calibrator._calibrate_paper` | Shared nonnegative slope and independent A/L intercepts, `sum_A omega/mean(omega) BCE + sum_L BCE`, no ridge |
| Label smoothing | `_calibrate_paper` | Hard labels on A, Platt targets on L using L's class counts |
| Weighted balance | `fit_intercept_shift` | A-only first-order intercept equation. With zero-positive A, fit L alone and anchor to the0.15 CP heuristic. Stratum IDs are preserved even when weights coincide |
| Calibration failure | `run_learned_calibration` | Keep refined labels and all verified labels, no extra sampling rounds in paper mode |
| Routing | `budgeted_calibrated_router.py` | Expected TP/FP/FN, alpha0.5 objective, opposing prefixes and Dinkelbach search |

`tests/test_paper_defaults.py` compares calibration to an independent numerical
objective and checks allocation and batch behavior. `tests/test_manuscript_graph_workload.py`
checks the graph equations, refinement phases and Cars Q7 workload. Other tests
cover the sampler, routing objective, oracle failures and public Spark operators.
