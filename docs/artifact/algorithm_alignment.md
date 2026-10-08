# Algorithm alignment

Authority: manuscript worktree at `a48d421`, with 41 input hashes in `manuscript_snapshot.json`. The disabled appendix is separate. Since `815f9fb`, evaluation prose, the best-point table and tradeoff PDF changed; algorithm formulas did not.

| Paper requirement | Effective implementation | Change/evidence |
| --- | --- | --- |
| Fixed predicate-aware axis and same-pass hidden state | `ce_ranker.score_filter/score_join`, `proxy_model._reduce_rows` | Public default CE; skip unused SigLIP pass. Qwen revision pinned to `4bd860ac4f15ad1897a214615cccc700f8f71818`. Named columns and original row/pair IDs stay aligned. Cache includes model/content/backend identity. CE tests and real Spark smoke. |
| L2 then PCA256 | `_reduce_rows` | L2 before PCA and after projection; rank min(256,d,N−1). N<=1 retains at most256 normalized coordinates instead of fitting degenerate PCA. Population features, no labels, fit PCA. |
| Preparation budget | `paper_calibration_config`, budget properties, `PreparationLedger` | Exact integer split `(4B)//5`, `B-B_L`, including B0/1; no ten-label floor. D/fallback outside allowance. Tests cover B0..20, caching and failures. |
| Algorithms1–2 / Eq. gate | `SamplingConfig`, `_paper_sas`, `one_sided_lower` | Default paper/uniform/.3/n_min2. Strict lower-bound >.05, permanent branch, no pilot batch minimum. Legacy tests explicitly select legacy. |
| SAS branches | `sampling_strata`, `threshold_search_multi`, `_paper_sas` | Geometric strata, bounded quotas, pilot Neyman/minimum exploration; carried .25 exploration, scan until2 positives, DP penalty1.5/max6, windows and seed service. Exact bounds/quota/trace tests. Population quality remains experimental. |
| S3 missing either class | `_whole_query_llm_fallback` | Retain known labels; skip later stages; uncapped stable-ID fallback. Real B2 filter/join each made4 unique calls: L1/fallback3, no failures. |
| Balanced L2 LR C1 | LR fit/refinement, public operator flags | Paper harness sets balanced weights/C1. Sampler-ablation tests verify LR-only behavior. |
| Gaussian symmetric graph and row-normalized P | `build_knn_graph`, `predict_with_score_refinement` | Residual pinning, label pinning, damping .8/.8, iterations50/50, threshold .5. Fold refinement excludes held-out anchors. |
| kNN conflict | `auto_knn_k` | Active table round(3log2N); disabled appendix cap5/50. Default follows active body, bounded by N−1. Historical cap requires `PROBE_KNN_PROTOCOL=legacy`. Historical experiment provenance must be reconciled. |
| Refined-score calibration strata | `run_learned_calibration`, `StratifiedDesign` | A uses refined probability, not LR probability. Merge to at most min(10,ceil(.2m)) strata so the even share samples every stratum. Within-stratum SRSWOR, weights N_j/n_j, remaining Neyman share. |
| Cross-fitting | `compute_oof_features`, `refine_fn` | min(5,n_pos,n_neg) folds; LR/refinement rebuilt per fold. Missing OOF omits L loss; A-only fit needs observed class evidence. No zero-A CP shape from unavailable OOF. |
| Eq. platt/loss | `calibrate(paper=True)` | Nonnegative slope/intercept/L-only xi; h coefficient fixed0; only .5xi² ridge. Smoothed targets use pooled A/L class counts. Independent three-parameter reference optimization agrees. |
| Eq. anchor / failure | `fit_intercept_shift`, `stratified_positive_target` | A-only inverse-weighted anchor, omit xi at inference. CP .15/.05 is a heuristic. Failed fit preserves refined labels and verified L/A labels. |
| Eq. counts/objective/two-threshold | `expected_counts`, `plan_dual_threshold` | Refined labels, opposing prefixes, Dinkelbach, alpha .5; alpha0 no routing. Brute-force tests verify objective/counts. |
| Public defaults / CLI | public operators, `run.py`, `reproduce.py` | Paper SAS and learned calibration resolve preparation B; explicit old/ablation decisions remain. Dry-run records protocol, revisions, workload N and seeds. |

See `validation.md`. Contracts and smoke do not establish paper quality/speedups. Frozen-L ablation and seven-stage timing remain experiment-level gaps in `experiment_alignment.md`.
