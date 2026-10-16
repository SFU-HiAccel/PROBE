# Changelog

## [Unreleased] - 2026-10-16

### Features
- Align preparation with the current manuscript: 60% training labels and 40% calibration labels, with routing outside that allowance.
- Resolve the default SAS batch cap from 1% of the candidate population.
- Fit the shared-slope Platt model with independent training/calibration intercepts, hard calibration labels and training-only Platt targets.
- Preserve up to ten calibration strata and reserve at least one draw in every stratum. Keep stratum identities distinct in the zero-positive CP heuristic.
- Use unit-normalized Gaussian union graphs and pin expert labels after both refinement phases.
- Add the image-only Cars Q7 case study and use embedding similarity for LOTUS joins.
- Correct Animals Q1 to 1,600 physical image rows and binary F1. Preserve row IDs through evaluation.
- Export scorer-aligned evaluation arrays, paired bootstrap intervals, ECE and all ten experiment figure/table families.
- Bind resume and reporting to input, code, model and annotation identities.
- Remove dynamic reranking and hybrid-pilot samplers, the learned-deferral router, the unused Spark JAR, unrelated workloads, workbook plots and obsolete environment snapshots.

### Design Rationale
- The current manuscript worktree, identified by file hashes, determines the released algorithm and workload configuration.
- Use the same expert messages, candidate IDs and transport across systems. Keep human annotations in evaluation code.
- Preserve measured outputs and reject mismatched protocols rather than combining runs from different algorithm revisions.

### Notes & Caveats
- Numerical outputs from the October 8 protocol are incompatible with this revision.
- Complete benchmark grids and paper-level statistics still require measured predictions, human annotations and configured model servers.
- Older research implementations remain available in Git history.
