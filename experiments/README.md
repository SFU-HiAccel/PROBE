# Experiments

The current VLDB plans live in `paper_manifest.json`. Run `python experiments/reproduce.py --experiment all --dry-run` to print all configurations without importing Spark or making oracle requests.

`reproduce.py` manages serial, locked jobs with immutable records and job-level resume. `report.py` validates raw output hashes, protocol, universe and seed coverage; it refuses full plotting from missing evidence. See [ARTIFACT.md](../docs/artifact/ARTIFACT.md) and [experiment_alignment.md](../docs/artifact/experiment_alignment.md) for executable arms and precise gaps.

The numbered `01_...sh` through `09_...sh` scripts, `PLAN.md`, `PLAN_PROMPT.md`, and root `paper_figures/` are historical research entry points. `common.sh` explicitly selects their legacy sampler/representation/calibration settings. Their results are not current-paper evidence merely because the scripts run.
