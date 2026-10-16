# Experiments

Run `reproduce.py` from the repository root. `paper_manifest.json` defines the
workloads, parameter grids and current manuscript protocol.

```bash
python experiments/reproduce.py --experiment all --dry-run
python experiments/reproduce.py --experiment main --systems PROBE \
  --queries match113 --seeds 0 --model-config models.local.json \
  --model-revision "$PROBE_ORACLE_REVISION" --output results/paper --resume
python experiments/report.py --input results/paper --output results/figures
python -m experiments.paper_statistics --input results/paper --output results/figures/bootstrap.json
python experiments/report.py --input results/paper --output results/figures \
  --plot-all --bootstrap results/figures/bootstrap.json
```

The root [README](../README.md#experiments) lists commands for every experiment.
`paper_pipeline.py` executes PROBE ablations, `paper_baselines.py` and
`scaledoc_adapter.py` execute baseline adapters, and `paper_statistics.py` implements
the paired seed/class-stratified bootstrap. Benchmark scoring is in `run.py`.

Jobs use full paper candidate populations, serial execution, per-job locks and
hashed outputs. Interrupted jobs restart in a new attempt directory. Reporting
rejects corrupt outputs, mixed protocols and duplicate seeds. Missing grid points
are recorded in `coverage.json`.

`--plots` selects individual complete experiment grids. Each renderer writes PDF/PNG
figures or CSV/LaTeX tables. Main intervals are tied to the validated completed records.
