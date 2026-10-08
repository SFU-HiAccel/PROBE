# PROBE

PROBE runs semantic filters and joins with a multimodal LLM, using fixed cross-encoder ranking, selective sampling, graph refinement and calibrated routing to reduce oracle calls.

This branch prepares the VLDB artifact. Public defaults follow the active manuscript in [manuscript_snapshot.json](docs/artifact/manuscript_snapshot.json). Algorithm tests and small real-model Spark examples pass. Full paper reproduction is **not yet verified**; [experiment_alignment.md](docs/artifact/experiment_alignment.md) records missing evidence and adapters.

## Start here

Use Python 3.12 and Java 11 for the tested Spark client. Install the client separately from the vLLM server:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.client.txt
export JAVA_HOME=/path/to/jdk11
export SPARK_LOCAL_IP=127.0.0.1
python experiments/smoke.py --dry-run
python experiments/reproduce.py --experiment main --systems PROBE --dry-run
```

Dry-run uses only the standard library and makes no model or oracle requests. For CPU checks, install `requirements.core.txt` and run:

```bash
python -m pytest -q tests/test_fixed_ce_paper_protocol.py tests/test_paper_defaults.py \
  tests/test_weighted_calibrator.py tests/test_stratified_calibration.py \
  tests/test_budgeted_calibrated_router.py tests/test_artifact_reporting.py
```

With an idle Gemma 4 server and GPU memory for the cross-encoder:

```bash
export PROBE_ORACLE_URL=http://127.0.0.1:8000/v1/chat/completions
python experiments/smoke.py --device cuda:0 --output results/my-smoke
```

The four-candidate filter and join use real models and oracle requests. A two-label preparation budget triggers S3: retain known labels and send remaining candidates to the expert. This checks integration, not paper quality.

## Code organization

| Component | Files |
| --- | --- |
| Public Spark operators | `sem_ops_register.py`, `proxy_model.py` |
| Fixed ranking/features | `ce_ranker.py`, `ce_ranker_backend.py` |
| SAS | `fixed_ce_sampling.py`, `sampling_strata.py`, `threshold_search_multi.py` |
| LR and graph refinement | `cluster_proxy.py` |
| Calibration/routing | `learned_calibration.py`, `stratified_calibration.py`, `weighted_calibrator.py`, `budgeted_calibrated_router.py` |
| Candidate/oracle accounting | `candidate_identity.py`, `oracle_budget.py` |
| Paper runs and validation | `experiments/reproduce.py`, `experiments/report.py`, `experiments/paper_manifest.json` |
| Benchmark loaders/scorers | `run.py` |

Preparation budget B splits into floor(0.8B) training labels and the remainder for calibration. Routing and S3 fallback are outside B. Reports separate unique labels, physical requests, GT quality and the router's estimated agreement with the expert.

See [ARTIFACT.md](docs/artifact/ARTIFACT.md) for installation and experiments, [algorithm_alignment.md](docs/artifact/algorithm_alignment.md) for formula-to-code mappings, and [validation.md](docs/artifact/validation.md) for measured coverage. Numbered shell scripts and old figure scripts are historical entry points with explicit legacy configuration.

The existing [MIT license](LICENSE) is preserved. Vendored systems, data and models keep upstream terms; see [NOTICE.md](NOTICE.md).
