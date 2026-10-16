# Reproduction notes

See the root [README](../../README.md) for installation, model configuration,
workloads and experiment commands.

- [Algorithm mapping](algorithm_alignment.md): manuscript equations and code.
- [Experiment alignment](experiment_alignment.md): grids, baseline adapters and remaining evidence gaps.
- [Manuscript snapshot](manuscript_snapshot.json): hashes of the authoritative source files.
- [Validation](validation.md): recorded checks and their source revisions.

The current protocol uses 60% of the preparation budget for training. Results from
older 80% configurations are incompatible with the new manifest. Raw expert labels,
human benchmark annotations and independently declared model revisions have distinct
roles and must remain distinguishable in result records.

Numerical reproducibility requires the complete measured grids. Model endpoints,
licensed input data and matching hardware/runtime settings are supplied by the
operator. The repository does not substitute historical workbook values for raw runs.
