# Experiment alignment and evidence gaps

`experiments/paper_manifest.json` is the machine-readable authority for active labels, full candidate universes, models/revisions, budgets, seeds, baseline protocols, command plans and evidence state. The manuscript snapshot is `a48d421`. No historical run has been promoted into current-protocol raw evidence.

| Active label | Named command (`python experiments/reproduce.py --experiment NAME --dry-run`) | State and next evidence |
| --- | --- | --- |
| fig:sys-f1, fig:sys-latency | main | Five seeds0..4,14 queries, B10/alpha.5 or target.9. New PROBE/no-proxy/BARGAIN executable wiring exists; complete grid and controlled timing missing. AIQA/CSV/LOTUS/ScaleDoc integrations require audit. |
| fig:sys-tradeoff, tab:main-comparison | sweep | Budget5/10/15/20/25, targets.1/.3/.5/.7/.9. Full same-universe runs missing. Quality-first minimizes delegation at max quality; cost-first requires90% model-matched no-proxy quality, with EcommQ9 closest-point exception. No manual point insertion. |
| fig:ablation sampling | sampling | SAS/uniform/Neyman-stratified/top-K share CE features and balanced LR. Formal execution blocked until B_L and zero-class behavior are reconciled consistently across arms; public LR-only contracts are tested. |
| fig:ablation refinement | refinement | Requires one frozen SAS label set and proxy across LR/bias/full. Independent stochastic runs cannot certify this. A dedicated frozen-L executor is missing. |
| fig:alpha-tradeoff | calibration | Platt/GAM/none, alpha0/.1/.3/.5/.7/.9/1. Current workbook Fig9_Alpha has13 queries, mixed seed0/1, edited aggregate cells, and an older no-calibration description. Do not certify it as complete14-query raw evidence. BARGAIN-A/P/R use separate target grids. |
| fig:repr-models | representation | Qwen CE/Jina CE/SigLIP/CLIP-base must change both axis and features. Model backend helpers exist; formal arm wiring/provenance remains unaudited. |
| fig:mllm-generalization | experts | Four separate model-matched expert/no-proxy pairs; immutable model revisions and archived server settings needed. The runner does not replace a live server. |
| fig:time-stack | timing | Paper requires seven measured stages. Existing four-phase accounting overlaps oracle/client phases; cannot certify this panel until reconciled. Delegation-scaled latency is excluded. |
| tab:lowsel-results | low_selectivity | CarsQ4 N157269, EcommQ6 N9280, MovieQ5 N65024, B10/25. Full measured records and ScaleDoc adapter pending; reject alternate research scale factors. |
| fig:encoder-rank, fig:sas-multi-process | trace | EcommQ9 N25440; figure-only batch300 distinct from batch32 default. Dedicated trace export/plot audit pending. |

Formal dry-runs are available for every named experiment. A plan is not proof of an executable or completed experiment. Unsupported arms stop before oracle calls. Reporting writes missing query/system coverage and refuses complete-paper plotting from partial inputs. Current generic diagnostic plotting is not a substitute for audited paper figure generators.

## Provenance observations

The source workbook `../PROBE-VLDB/data/latest_paper_data.xlsx` defines the current presentation, not the experimental facts. Its Fig9_Alpha description excludes CarsQ8, substitutes EcommQ13 seed1 for seed0 and contains edited string/aggregate cells. Its `T2_Latency_Simulation` sheet and historical figure scripts include estimated quantities. Those values cannot certify measured end-to-end latency. `experiment_sources.json` records the workbook hash and the exact source descriptions audited without copying them into new raw runs.

The active paper's default geometric-mean claims are PROBE/no-proxy F1 .762/.797, PROBE delegation .128, speedup4.88; baseline F1 .530/.628/.484 and median delegation .81/.21/.80. Each is listed as unverified raw evidence in the manifest. The latest cost-first prose now uses44.0s/4.9x; these are also unverified. No raw value is synthesized to meet a paper number.

Main quality requires five actual seeds for each exact variant and query. A single seed, 13-query subset, alternative candidate scale, GT replay, or variance from another variant is not interchangeable. Semantic-set scores and Animals count-derived quality remain explicitly named, distinct from binary candidate F1 and calibration against the expert. Missing queries remain missing; zeros are retained in geometric means.

The appendix input is disabled. Historical sensitivity/GAMCAL/report scripts are retained with historical identity and are not part of active acceptance. Missing original SemBench vendor revision, complete input checksums and expert server weight revision remain release provenance gaps. Large/full-grid reruns need those prerequisites and audited adapters before they can be claimed reproducible.
