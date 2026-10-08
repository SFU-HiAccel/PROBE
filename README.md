# PROBE: Learned Proxy-Based Execution for Highly Selective Multimodal Semantic Queries

This repository contains the implementation and experiment entry points for the paper
**“PROBE: Learned Proxy-Based Execution for Highly Selective Multimodal Semantic Queries.”**
PROBE stands for learned **Pro**xy-**B**ased **E**xecution. It evaluates semantic SQL
filters and joins over text and images while reducing calls to an expensive expert
multimodal large language model (MLLM).

The `vldb-artifact` branch is the default branch. This guide describes the paper's
components, workloads and experiment sections. The artifact's recorded manuscript
snapshot is identified by source hashes
in [manuscript_snapshot.json](docs/artifact/manuscript_snapshot.json), with source HEAD
`a48d421`; later manuscript edits are not automatically covered by that snapshot.

**Reproduction status:** the algorithm contracts and small real-model Spark examples
have been checked. Complete reproduction of the paper's results is still pending.
The instructions below distinguish executable runs from plans whose execution
adapters are missing. See [validation.md](docs/artifact/validation.md) for measured
evidence and [experiment_alignment.md](docs/artifact/experiment_alignment.md) for gaps.

- [How It Works](#how-it-works)
- [The Three Components](#the-three-components)
- [Repository Layout](#repository-layout)
- [Workload](#workload)
- [Installation](#installation)
- [Data and Models](#data-and-models)
- [Run the Experiments](#run-the-experiments)
- [Generate the Figures and Tables](#generate-the-figures-and-tables)
- [Reproducing the Baselines](#reproducing-the-baselines)
- [Validation and Troubleshooting](#validation-and-troubleshooting)
- [License](#license)

## How It Works

```text
PySpark DataFrame: filter rows or semantic-join candidate pairs
  │  df.sem_filter(...) / df.sem_join(...)
  ▼
Fixed cross-encoder pass: ranking scores and row features
  ▼
① Selectivity-Adaptive Sampling → expert labels L → logistic-regression proxy
  ▼
② Graph-Based Proxy Refinement → refined scores for all candidates
  ▼
③ Learned Calibrated Cascade Routing
    calibration sample A → calibrated probabilities → routed expert labels D
  ▼
Filtered / joined PySpark DataFrame
```

The default representation is **Qwen3-VL-Reranker-2B**. One encoder pass provides
both the ordering scores used by sampling and the features used by proxy training
and graph construction. Paper runs use the full candidate universe listed below;
the formal runner disables join blocking and rejects a candidate count different
from the paper's count.

The user controls preparation budget `B` (a percentage of candidates) and routing
tradeoff `alpha` (default `0.5`). For `N` candidates and budget percentage `b`, the
preparation allowance is `B = floor(N * b / 100)`, with `B_L = floor(0.8 * B)` training
labels and `B_A = B - B_L` calibration labels. Routing and whole-query fallback can
make additional expert calls outside `B`. Outputs record unique labeled candidates
and physical requests separately.

## The Three Components

### 1. Selectivity-Adaptive Sampling — paper Section 3

[fixed_ce_sampling.py](fixed_ce_sampling.py) begins with a uniform pilot using
`30%` of `B_L`. If its selectivity test indicates a less selective query, it uses
stratified sampling. Otherwise, it explores and exploits multiple positive intervals
on the fixed encoder ranking, implemented in
[threshold_search_multi.py](threshold_search_multi.py). The labeled set trains a
class-balanced logistic-regression proxy. If training labels contain only one class,
safeguard S3 sends remaining candidates to the expert and retains known labels.

### 2. Graph-Based Proxy Refinement — paper Section 4

[cluster_proxy.py](cluster_proxy.py) constructs a nearest-neighbor graph over the
encoder features and refines proxy scores in two phases: **bias diffusion** and
**homogeneity propagation**. Neither phase requests new expert labels. Defaults are
`k = min(N - 1, round(3 * log2(N)))`, damping `0.8` for both phases, up to `50`
iterations per phase, and refined-score threshold `0.5`.

### 3. Learned Calibrated Cascade Routing — paper Section 5

[learned_calibration.py](learned_calibration.py) draws calibration sample `A` with
known inclusion probabilities and fits a Platt-style calibrator using refined score
logits. Out-of-fold training scores inform the slope; inverse-probability weighted
calibration labels anchor the level.
[budgeted_calibrated_router.py](budgeted_calibrated_router.py) selects rows to send
to the expert by minimizing `alpha * (1 - estimated F1) + (1 - alpha) * delegation`.
Estimated agreement with the expert and benchmark quality against human ground
truth are separate quantities.

See [algorithm_alignment.md](docs/artifact/algorithm_alignment.md) for formula-to-code
mappings and safeguard details.

## Repository Layout

| File or directory | Purpose |
| --- | --- |
| [sem_ops_register.py](sem_ops_register.py), [proxy_model.py](proxy_model.py) | Public Spark semantic operators and proxy entry points |
| [ce_ranker.py](ce_ranker.py), [ce_ranker_backend.py](ce_ranker_backend.py) | Fixed cross-encoder scores, features, provenance and caches |
| [fixed_ce_sampling.py](fixed_ce_sampling.py), [sampling_strata.py](sampling_strata.py), [threshold_search_multi.py](threshold_search_multi.py) | Selectivity-Adaptive Sampling and multi-interval search |
| [cluster_proxy.py](cluster_proxy.py) | Logistic-regression proxy and graph refinement |
| [learned_calibration.py](learned_calibration.py), [stratified_calibration.py](stratified_calibration.py), [weighted_calibrator.py](weighted_calibrator.py), [budgeted_calibrated_router.py](budgeted_calibrated_router.py) | Calibration sampling, fitting and cascade routing |
| [candidate_identity.py](candidate_identity.py), [oracle_budget.py](oracle_budget.py) | Stable candidate IDs, expert transport and call accounting |
| [run.py](run.py) | Benchmark loaders, prompts, query scorers and execution |
| [experiments/reproduce.py](experiments/reproduce.py) | Named paper plans, serial execution, run locking and resume |
| [experiments/paper_manifest.json](experiments/paper_manifest.json) | Workload counts, protocols, systems, seeds, grids and execution availability |
| [experiments/report.py](experiments/report.py) | Raw-output validation, aggregation and coverage reports |
| [experiments/smoke.py](experiments/smoke.py) | Small real-model Spark filter/join integration check |
| [docs/artifact/](docs/artifact/) | Environment, algorithm alignment, experiment gaps and validation records |
| [tests/](tests/) | Numerical, public-operator, oracle-accounting and reporting checks |
| [third_party/](third_party/) | Vendored baseline and benchmark sources; upstream licenses apply |
| [paper_figures/](paper_figures/) | Historical workbook-based plot scripts and outputs |

## Workload

The main evaluation uses **14 queries** from SemBench and LRO-Bench. `F` denotes
a semantic filter and `J` a semantic join. Use the exact query IDs in commands.

| Query ID | Scenario | Type | Modality | Candidates `N` | Positive rate |
| --- | --- | :---: | --- | ---: | ---: |
| `mmqa_q2a` | MMQA | J | text + image | 10,400 | 5.0e-4 |
| `mmqa_q7` | MMQA | J | text + image | 230,400 | 2.17e-5 |
| `ecomm_q7` | E-commerce | J | text | 525,625 | 9.4e-3 |
| `ecomm_q8` | E-commerce | J | text | 115,768 | 3.0e-4 |
| `ecomm_q9` | E-commerce | J | text + image | 25,440 | 2.65e-2 |
| `ecomm_q13` | E-commerce | F | text + image | 3,992 | 3.0e-3 |
| `cars_q3` | Cars | F | text + image | 10,070 | 3.39e-2 |
| `cars_q4` | Cars | F | text | 157,269 | 1.39e-1 |
| `cars_q8` | Cars | F | text + image | 30,130 | 5.4e-3 |
| `animals_q1` | Animals | F | text + image | 1,600 | 6.1e-2 |
| `match113` | LRO / Santos | J | text | 266 | 7.52e-3 |
| `match121` | LRO / Santos | J | text | 1,426 | 4.91e-3 |
| `match202` | LRO / Movie | J | text | 384 | 7.81e-3 |
| `match305` | LRO / Schools | J | text | 1,421 | 7.74e-3 |

The less selective case study also uses `ecomm_q6` (`N = 9,280`) and `movie_q5`
(`N = 65,024`), alongside `cars_q4`. The manifest selects required scenario scale
factors automatically, including Ecomm `sf_250` for the main workload and `sf_4000`
for Ecomm Q6.

Quality uses each benchmark's human-GT query scorer. Animals Q1 is an aggregate
count query, and Movie Q5 uses semantic-set scoring; their metrics must retain
those identities rather than being presented as binary candidate F1.

## Installation

### Hardware and software environment

The paper uses two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, a 20-core Intel Xeon
w7-3445 and 128 GB RAM. The validated client uses **Python 3.12 and Java 11**.
The Spark/encoder client and vLLM expert server use separate environments.
Client dependencies are specified in [requirements.client.txt](requirements.client.txt),
with the recorded resolved environment in [requirements.client.lock.txt](requirements.client.lock.txt).
Observed validation environments are recorded in
[environment.json](docs/artifact/validation/environment.json).

### Install the client

```bash
git clone --branch vldb-artifact https://github.com/PROBEProject/PROBE.git
cd PROBE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.client.txt
export JAVA_HOME=/path/to/jdk11
export SPARK_LOCAL_IP=127.0.0.1

# Preview configurations without downloading models or invoking Spark/the expert.
python experiments/smoke.py --dry-run
python experiments/reproduce.py --experiment main --systems PROBE --dry-run
```

Dry-run uses only Python's standard library, so it can also be used before installing
the client dependencies. For numerical tests without Spark or model inference,
install [requirements.core.txt](requirements.core.txt) in a separate environment.
Older `create_*_env.sh` and `environment.*` files are historical snapshots; the
installation above is the validated artifact client path.

## Data and Models

### Benchmark data

Obtain [SemBench](https://github.com/SemBench/SemBench) and
[LRO-Bench](https://github.com/LROBench/LROBench) and follow their preparation
instructions, including images and human annotations. The loaders in `run.py`
expect the following layout; preserve benchmark filenames:

```text
data-root/
  SemBench/files/
    cars/data/sf_157376/...
    cars/raw_results/ground_truth/...
    ecomm/data/sf_250/...
    ecomm/data/sf_4000/...
    ecomm/raw_results/ground_truth/...
    ecomm/raw_results/ground_truth_sf4000/...
    mmqa/...
    animals/...
    movie/...
  databases/
    santos/...
    movies/...
    california_schools/...
```

```bash
export PROBE_DATA_ROOT=/path/to/data-root
export LROBENCH_DATABASES_DIR="$PROBE_DATA_ROOT/databases"
```

`--data-root` and `--lro-root` can override these locations. The default data root
is `data/`. Full input checksums and the original SemBench vendor revision remain
provenance gaps, documented in [ARTIFACT.md](docs/artifact/ARTIFACT.md).

### Representation model and expert server

| Role | Default model | Revision handling |
| --- | --- | --- |
| Ranking and proxy features | [Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | Pinned to `4bd860ac4f15ad1897a214615cccc700f8f71818`; downloaded through Hugging Face |
| Expert MLLM | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | Operator supplies the immutable weight revision used by the server |

Serve the expert separately through a vLLM OpenAI-compatible chat endpoint, with
the model's access terms satisfied. Record its weight revision, server version,
batching and cache settings. The runner connects to that server; it does not start
or switch it. A model name alone does not verify the weight revision.

```bash
export PROBE_ORACLE_URL=http://127.0.0.1:8000/v1/chat/completions
export PROBE_ORACLE_REVISION=YOUR_IMMUTABLE_EXPERT_REVISION

# Requires the real expert server and GPU memory for the cross-encoder.
python experiments/smoke.py --device cuda:0 --output results/my-smoke
```

Use a new smoke output directory. This runs a four-candidate filter and a four-pair
join with real models; its small preparation budget exercises S3 fallback. It
validates integration and does not measure paper quality or speedup.

## Run the Experiments

### Paper section → experiment → command

Run commands from the repository root. The following commands **preview** each
experiment's resolved grid. Section names and LaTeX labels identify the paper
artifacts independently of figure numbering.

| Paper section and artifact | Experiment configuration | Preview command | Execution status |
| --- | --- | --- | --- |
| End-to-end Comparison: `fig:sys-f1`, `fig:sys-latency` | 14 queries; seeds 0–4; preparation 10%, PROBE alpha 0.5, baseline target 0.9 | `python experiments/reproduce.py --experiment main --dry-run` | PROBE, no-proxy and BARGAIN-A executable; remaining baseline adapters pending |
| Quality–Cost Tradeoff: `fig:sys-tradeoff`, `tab:main-comparison` | Budgets 5/10/15/20/25%; target grid 0.1/0.3/0.5/0.7/0.9 | `python experiments/reproduce.py --experiment sweep --dry-run` | Same three systems executable; full baseline grid and best-point table reconstruction pending |
| Ablation on Sampling Policy: `fig:ablation` (a) | SAS / uniform / Neyman-stratified / top-K; same CE features and balanced LR; refinement/calibration/routing off | `python experiments/reproduce.py --experiment sampling --dry-run` | Plan only; consistent `B_L` execution and paper's 5–25% sweep still needed; current plan contains 10% |
| Ablation on Graph Refinement Stage: `fig:ablation` (b) | LR / bias diffusion / full refinement, sharing one frozen SAS label set | `python experiments/reproduce.py --experiment refinement --dry-run` | Plan only; frozen-label executor and full budget sweep missing; current plan contains 10% |
| Ablation on Learned Calibrated Cascade Routing: `fig:alpha-tradeoff` | Platt / GAM / none; alpha 0/0.1/0.3/0.5/0.7/0.9/1; BARGAIN target sweep | `python experiments/reproduce.py --experiment calibration --dry-run` | PROBE and BARGAIN-A/P/R executable; complete measured coverage and calibration-error reporting pending |
| Ablation on Representation Model: `fig:repr-models` | Qwen CE / Jina CE / SigLIP / CLIP-base, replacing ordering scores and proxy features | `python experiments/reproduce.py --experiment representation --dry-run` | Plan only; alternative-model execution/provenance audit pending |
| Ablation on the Expert LLM: `fig:mllm-generalization` | Qwen3-VL-30B-A3B-Instruct / Qwen3.8-27B / Gemma4-26B-A4B / GLM-4.6V-Flash; matched no-proxy per model | `python experiments/reproduce.py --experiment experts --dry-run` | Plan only; separate endpoints, revisions and expert-arm wiring needed |
| Time Breakdown: `fig:time-stack` | Seven measured stages: embedding, sampling, expert calls, calibration, refinement, proxy training, proxy prediction | `python experiments/reproduce.py --experiment timing --dry-run` | Plan only; non-overlapping seven-stage accounting missing |
| Case Study: Less Selective Queries: `tab:lowsel-results` | Cars Q4 / Ecomm Q6 / Movie Q5; preparation 10% and 25%; alpha 0.5, baseline target 0.9 | `python experiments/reproduce.py --experiment low_selectivity --dry-run` | PROBE, no-proxy and BARGAIN-A executable; ScaleDoc adapter pending |
| Sampling illustrations: `fig:encoder-rank`, `fig:sas-multi-process` | Ecomm Q9; figure-only batch size 300, separate from default batch size 32 | `python experiments/reproduce.py --experiment trace --dry-run` | Plan only; trace exporter and plotting audit pending |

```bash
# Preview all ten named plans. This does not execute the experiments.
python experiments/reproduce.py --experiment all --dry-run
```

An executable arm means the runner is wired to the implementation; it does not mean
the complete paper grid has been rerun or certified. Without `--dry-run`, plans with
unavailable adapters stop before expert calls. For example, the default `main` plan
includes all baselines, so select available systems to run the executable subset.

### Execute available experiments

First check one small real benchmark query:

```bash
python experiments/reproduce.py --experiment main --systems PROBE \
  --queries match113 --seeds 0 --device cuda:0 \
  --data-root "$PROBE_DATA_ROOT" --lro-root "$LROBENCH_DATABASES_DIR" \
  --model-revision "$PROBE_ORACLE_REVISION" \
  --output results/paper --resume
```

Then run available arms for the main comparison and other experiments:

```bash
# Main quality: all 14 queries, five seeds per system/configuration.
python experiments/reproduce.py --experiment main \
  --systems PROBE no_proxy BARGAIN-A --device cuda:0 \
  --model-revision "$PROBE_ORACLE_REVISION" --output results/paper --resume

# Budget/target sweep: the manifest currently plans seed 0.
python experiments/reproduce.py --experiment sweep \
  --systems PROBE no_proxy BARGAIN-A --device cuda:0 \
  --model-revision "$PROBE_ORACLE_REVISION" --output results/paper --resume

# Calibration/routing comparison, including BARGAIN modes.
python experiments/reproduce.py --experiment calibration \
  --systems PROBE BARGAIN-A BARGAIN-P BARGAIN-R --device cuda:0 \
  --model-revision "$PROBE_ORACLE_REVISION" --output results/paper --resume

# Less selective case study: three queries at two planned budgets.
python experiments/reproduce.py --experiment low_selectivity \
  --systems PROBE no_proxy BARGAIN-A --device cuda:0 \
  --model-revision "$PROBE_ORACLE_REVISION" --output results/paper --resume
```

These commands inherit the data paths and expert endpoint exported above. Use
`--queries`, `--systems` and `--seeds` to select a subset, and preview the same
arguments with `--dry-run` before launching. A subset remains a partial reproduction.
Full grids can require millions of expert calls. Run the client serially against
the same controlled server settings when comparing latency.

For experiments marked **plan only**, removing `--dry-run` is insufficient. Their
missing execution requirements are listed above and in
[experiment_alignment.md](docs/artifact/experiment_alignment.md). Older
`experiments/01_...sh` through `09_...sh` scripts explicitly use legacy settings
and do not fill these current-paper gaps. The appendix is disabled in the recorded
manuscript; historical hyperparameter and appendix scripts are outside this guide's
active reproduction target.

### Outputs and resume

Each job receives a directory derived from configuration, source hashes and data paths:

```text
results/paper/<run-id>/
  completed.json                 # job identity, measured row and output hashes
  attempt-001/
    resolved.json                # source hashes and resolved job configuration
    stdout.log                   # Spark/client execution log
    oracle.jsonl                 # actual expert responses and call stages
    <query>_results.csv           # query metrics and profiling
    *_result.csv/part-*.csv       # raw Spark prediction output
```

`--resume` skips intact completed jobs. Interrupted jobs restart in a new preserved
attempt, which may repeat expert requests; it is not recovery inside an oracle
batch. Existing completed outputs are not overwritten. The reporter checks raw
output hashes, candidate counts, human-GT provenance and expert failures.

## Generate the Figures and Tables

### Validate and aggregate new runs

```bash
python experiments/report.py --input results/paper --output results/figures
```

| Output | Contents |
| --- | --- |
| `validated_runs.json` / `.csv` | Individual measurements accepted by validation checks; CSV is written when records exist |
| `aggregate.json` / `.csv` | Per-query/configuration means, observed seeds and completeness; CSV is written when groups exist; main F1 has a Student-t 95% CI only with all five seeds |
| `coverage.json` | Missing query/system/configuration groups for each paper experiment |
| `recomputed_claims.json` | Main-evaluation aggregates for systems with complete seed and query coverage |

Zeros remain in geometric means; missing queries are reported rather than silently
dropped. The tradeoff table requires selecting points from complete measured sweeps:
quality-first minimizes delegation among maximum-quality points; cost-first minimizes
delegation among points reaching 90% of model-matched no-proxy quality, with the
paper's stated Ecomm Q9 exception.

```bash
# Currently refuses incomplete experiment coverage.
python experiments/report.py --input results/paper --output results/figures --plot-all
```

Even after coverage checks pass, `--plot-all` produces
`<experiment>_quality_delegation_diagnostic.pdf` files. Exact paper-layout quality,
speedup, calibration, representation, timing and trace figures, and the best-point
LaTeX table, still require audited figure/table generators. The manifest's planned
PDF names are output targets, not evidence that those files have been reproduced.

### Historical figures

[paper_figures/make_figures.sh](paper_figures/make_figures.sh) redraws older figures
from `paper_figures/plotdata.xlsx`:

```bash
# Historical workbook rendering only; does not rerun current paper experiments.
bash paper_figures/make_figures.sh
```

It writes PDFs/PNGs under `paper_figures/figures/` and LaTeX tables under
`paper_figures/tables/`. Existing workbook cells and PDFs are presentation artifacts;
they do not replace current-protocol raw runs. Their older model lists, router
variants and estimated timing must not certify the current paper.

## Reproducing the Baselines

All comparisons must use the same expert, prompts, full candidate universe and
human-GT scorer. Select available systems with `--systems` in the commands above.

| System | Paper configuration | Current reproduction entry/status |
| --- | --- | --- |
| `no_proxy` | Expert evaluates every candidate with all proxy optimizations disabled | Executable in `main`, `sweep`, `low_selectivity`; same stable-ID expert transport |
| `BARGAIN-A` | PROBE's refined proxy; accuracy target 0.9, failure probability 0.1 | Executable in `main`, `sweep`, `calibration`, `low_selectivity`; vendored implementation via [bargain_refined.py](bargain_refined.py) |
| `BARGAIN-P`, `BARGAIN-R` | Same refined proxy; precision/recall target sweep; uncapped expert budget | Executable in `calibration`; use that experiment's target grid |
| `AIQA` | UQE active sampling and its learned proxy | Named plan available; current-paper representation/integration audit pending |
| `CSV` | Cluster counts 4 / 2; 10% samples per cluster; 90% label consensus | Named plan available; current-paper adapter audit pending |
| `LOTUS` | Qwen3-VL-2B token-logprob proxy; precision/recall targets 0.9, failure probability 0.1, sampling 10% | Named plan available; current-paper adapter audit pending |
| `ScaleDoc` | Frozen embeddings, contrastive MLP, training 5% / calibration 5%, F1 target 0.9 | Named plan available; current-paper adapter audit pending |

Older baseline scripts are retained for research context. A successful legacy
baseline run is not automatically comparable with the active manifest protocol.

## Validation and Troubleshooting

```bash
# Full client regression suite; requires client dependencies and Java 11.
python -m pytest -q tests

# Selected numerical/protocol checks, usable with requirements.core.txt.
python -m pytest -q tests/test_fixed_ce_paper_protocol.py tests/test_paper_defaults.py \
  tests/test_weighted_calibrator.py tests/test_stratified_calibration.py \
  tests/test_budgeted_calibrated_router.py tests/test_artifact_reporting.py
```

The recorded clean-clone validation passed **356 tests**. Real integration checks
covered the small Spark filter/join and one `match113` benchmark run. These checks
do not establish paper-wide quality or latency claims.

| Symptom | Check |
| --- | --- |
| Java gateway startup fails | `JAVA_HOME` points to an installed Java 11 runtime |
| Benchmark files are missing or `N` differs | Prepared data root, scenario scale factor and exact query universe |
| Expert endpoint/model error | Server availability, served model name, endpoint URL and weight revision |
| CE download or GPU-memory error | Hugging Face access/cache and memory for the pinned cross-encoder alongside the expert |
| `Unaudited adapters` | Select an executable system/experiment or inspect the documented missing adapter |
| Full plotting refuses to run | Read `coverage.json`; incomplete seeds, queries or systems remain partial evidence |

Further installation and provenance details are in [ARTIFACT.md](docs/artifact/ARTIFACT.md).

## License

PROBE retains its [MIT license](LICENSE). Vendored systems, benchmark data and model
weights retain their upstream terms; see [NOTICE.md](NOTICE.md).
