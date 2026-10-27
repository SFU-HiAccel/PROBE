# PROBE

Implementation of **PROBE: Learned Proxy-Based Execution for Highly Selective
Multimodal Semantic Queries**.

PROBE evaluates semantic filters and joins over text and images. A fixed
cross-encoder ranks candidates and supplies their features. Selectivity-adaptive
sampling acquires expert labels for a logistic-regression proxy, a nearest-neighbor
graph refines its scores, and a calibrated cascade selects further expert calls.


## Installation

The client requires Python 3.12 and Java 11. The expert runs in a separate vLLM
server. The paper's machine has two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs,
a 20-core Xeon w7-3445 and 128 GB RAM.

```bash
git clone https://github.com/SFU-HiAccel/PROBE.git
cd PROBE
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.client.txt
export JAVA_HOME=/path/to/jdk11
export SPARK_LOCAL_IP=127.0.0.1
```

[requirements.client.lock.txt](requirements.client.lock.txt) records the resolved
client packages. CPU algorithm tests need only `requirements.core.txt`.

## Data and models

Prepare [SemBench](https://github.com/SemBench/SemBench) and
[LRO-Bench](https://github.com/LROBench/LROBench), including images and human
annotations, using the upstream preparation instructions. Expected directories:

```text
data-root/
  SemBench/files/
    cars/data/sf_157376/
    cars/data/full_data/image_data_full.csv
    cars/raw_results/ground_truth/
    ecomm/data/sf_250/
    ecomm/data/sf_4000/
    ecomm/raw_results/ground_truth/
    ecomm/raw_results/ground_truth_sf4000/
    mmqa/
    animals/
    movie/
  databases/
    santos/
    movies/
    california_schools/
```

```bash
export PROBE_DATA_ROOT=/path/to/data-root
export LROBENCH_DATABASES_DIR="$PROBE_DATA_ROOT/databases"
export PROBE_ORACLE_URL=http://127.0.0.1:8000/v1/chat/completions
export PROBE_ORACLE_REVISION=YOUR_40_CHARACTER_WEIGHT_COMMIT
cp experiments/models.example.json models.local.json
```

Edit `models.local.json` with the served model IDs, endpoints and weight commits.
The default expert is `google/gemma-4-26B-A4B-it`. The representation is
`Qwen/Qwen3-VL-Reranker-2B`, pinned to
`4bd860ac4f15ad1897a214615cccc700f8f71818`. Alternate representations and all four
expert bindings are listed in the example. Model downloads precede measured runs.

LOTUS filters require a Qwen3-VL-2B endpoint with token log-probabilities. LOTUS
joins use E5 or CLIP embedding similarity. ScaleDoc requires its upstream source:

```bash
git clone https://github.com/Seurgul/ScaleDoc.git external/ScaleDoc
git -C external/ScaleDoc checkout 33d6e815a84f9ac295336d6ece97f66adab5fa51
export SCALEDOC_SOURCE="$PWD/external/ScaleDoc"
```

Keep the server's batching, caching and parallelism settings fixed for timing
comparisons. Record those settings and the loaded weight commits with the results.
The runner checks endpoint model IDs but cannot attest which weights the server loaded.

## Workloads

| Query | Operator | Input | Candidates |
| --- | --- | --- | ---: |
| `mmqa_q2a` | join | text + image | 10,400 |
| `mmqa_q7` | join | text + image | 230,400 |
| `ecomm_q7` | join | text | 525,625 |
| `ecomm_q8` | join | text + image | 115,768 |
| `ecomm_q9` | join | text + image | 25,440 |
| `ecomm_q13` | filter | text + image | 3,992 |
| `cars_q3` | filter | text + image | 10,070 |
| `cars_q4` | filter | text | 157,269 |
| `cars_q8` | filter | text + image | 30,130 |
| `animals_q1` | filter | text + image | 1,600 |
| `match113` | join | text | 266 |
| `match121` | join | text | 1,426 |
| `match202` | join | text | 384 |
| `match305` | join | text | 1,421 |

The less selective case study adds `cars_q7` (30,130 images, dented-image predicate),
`ecomm_q6` (9,280 candidates) and `movie_q5` (65,024 candidates). Cars Q7 evaluates
the image predicate alone, rather than SemBench's union of image, audio and text
predicates. Its scorer derives the image-only reference from human annotations.
Animals Q1 uses row-level F1 and Movie Q5 uses semantic-set quality.
Animals Q1 uses 1,600 annotated image rows with stable IDs, replacing the older
count-accuracy metric. Ecomm Q8 follows the benchmark's image–text predicate,
although the manuscript table lists it as text-only.

## Experiments

Preview the complete configuration grid without models or Spark:

```bash
python experiments/reproduce.py --experiment all --dry-run
```

Use a new output directory for an integration check with real models:

```bash
python experiments/smoke.py --device cuda:0 --output results/smoke
```

Define the following helper to execute paper experiments:

```bash
paper_run() {
  python experiments/reproduce.py --device cuda:0 \
    --model-revision "$PROBE_ORACLE_REVISION" \
    --model-config models.local.json --scaledoc-source "$SCALEDOC_SOURCE" \
    --output results/paper --resume "$@"
}

paper_run --experiment main --systems PROBE --queries match113 --seeds 0
```

| Experiment | Command | Settings |
| --- | --- | --- |
| End-to-end comparison | `paper_run --experiment main` | 14 queries, seven systems, five seeds, budget 10% or target 0.9 |
| Quality–cost tradeoff | `paper_run --experiment sweep` | Budgets 5/10/15/20/25%, targets 0.1/0.3/0.5/0.7/0.9 |
| Sampling policies | `paper_run --experiment sampling` | SAS, uniform, Neyman-stratified, top-K, LR only |
| Graph refinement | `paper_run --experiment refinement` | LR, bias diffusion, full refinement, shared training labels |
| Calibration and routing | `paper_run --experiment calibration` | Platt, GAM, uncalibrated, alpha sweep, BARGAIN A/P/R |
| Representation models | `paper_run --experiment representation` | Qwen CE, Jina CE, SigLIP, CLIP-base |
| Expert models | `paper_run --experiment experts` | Four experts with matching no-proxy references |
| Runtime breakdown | `paper_run --experiment timing` | Seven exclusive stages and separate client/Spark overhead |
| Less selective queries | `paper_run --experiment low_selectivity` | Cars Q7, Ecomm Q6, Movie Q5, budgets 10% and 25% |
| Sampling illustrations | `paper_run --experiment trace` | Ecomm Q9 ranking and sampling events, figure batch size 300 |

`--queries`, `--systems`, `--seeds`, `--budgets`, `--alphas`, `--representations`
and `--experts` select subsets. See [paper_manifest.json](experiments/paper_manifest.json)
for the full grids and [experiment_alignment.md](docs/artifact/experiment_alignment.md)
for baseline settings.

Sampling and refinement ablations use the training share `floor(0.6 * B)` and
retain purchased labels. A single-class sample produces a constant proxy in these
ablations because routing is disabled. `--ablation-single-class oracle` selects a
separate S3 variant. Full PROBE always retains S3 fallback.

Each job stores its configuration, source hashes, measured metrics, predictions,
expert-response journal, input checksums and stage timings in `results/paper/<run-id>/`.
Prepared SemBench and LRO input contents are hashed before resume checks.
`--resume` skips intact completed jobs with matching code, models and inputs. Interrupted jobs restart in a new attempt
directory, which can repeat expert calls. Runs with unresolved answers or a
candidate count different from the manifest fail validation.

```bash
python experiments/report.py --input results/paper --output results/figures
python -m experiments.paper_statistics --input results/paper \
  --output results/figures/bootstrap.json --replicates 10000
python experiments/report.py --input results/paper --output results/figures \
  --plot-all --bootstrap results/figures/bootstrap.json
```

The first command reports coverage even when a grid is incomplete. Bootstrap
requires all five seeds for every main query/system. Plotting requires complete
selected grids. Use `--plots sampling refinement` to render just those families.
Figures are written as PDF and PNG, and operating-point tables as CSV and LaTeX.
The bootstrap report includes per-query and geometric-mean intervals, paired
differences, and the subset with fewer than 20 positives.

Main and trace runs export predictions and annotations after measured execution.
Retrieval statistics use the benchmark's distinct result keys. Animals Q1 keeps
physical rows and evaluates their binary labels. The exported arrays must reproduce the query's
measured quality. Calibration experiments also export probabilities and ECE on
human annotations outside L and A, using ten equal-width bins. The paper does not
specify an ECE evaluation population or bin count, so these settings are recorded.

Reporting validates raw hashes, code/data/model/annotation identities, candidate
counts, paired acquisitions and seed coverage. Bootstrap artifacts are bound to
these same records. The repository provides the execution and analysis code.
The checked-in [validation record](docs/artifact/validation.md) does not establish
full numerical reproduction of the paper's reported results.

## Development

```bash
python -m pytest -q tests
```

Spark tests require Java 11. Core modules are `fixed_ce_sampling.py`,
`threshold_search_multi.py`, `cluster_proxy.py`, `learned_calibration.py`,
`weighted_calibrator.py` and `budgeted_calibrated_router.py`.
`sem_ops_register.py` exposes the PySpark operators, and `run.py` contains benchmark
loaders and scorers.

## License

PROBE is distributed under the [MIT License](LICENSE). Vendored dependencies retain
their upstream licenses. See [NOTICE.md](NOTICE.md).
