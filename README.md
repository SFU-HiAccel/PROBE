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
`c5b2d86`; later manuscript edits are not automatically covered by that snapshot.

**Reproduction status:** the algorithm contracts and small real-model Spark examples
have been checked. Complete reproduction of the paper's results is still pending.
All ten named experiment families have execution paths; the commands below select
the paper grids and require real benchmark data and model endpoints. See [validation.md](docs/artifact/validation.md) for measured
evidence and [experiment_alignment.md](docs/artifact/experiment_alignment.md) for gaps.

- [How It Works](#how-it-works)
- [The Three Components](#the-three-components)
- [Workload](#workload)
- [Installation](#installation)
- [Data and Models](#data-and-models)
- [Run the Experiments](#run-the-experiments)
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

### Configure the models

The runner executes the paper's ten experiment families through `run.py`, using
actual expert calls, the benchmark's human-GT scorer and the full candidate
universe. It rejects unresolved expert answers, missing GT provenance and a
candidate count that differs from the manifest.

Copy [models.example.json](experiments/models.example.json) to `models.local.json`.
Replace every `REPLACE_WITH_SERVED_WEIGHT_COMMIT` used by your selected arms with
the **40-character commit of the weights actually loaded by that server**, and
set its served model ID and URL. The expert ablation requires a separate binding
for each of the four experts; every no-proxy reference uses the matching binding.
LOTUS additionally requires a real Qwen3-VL-2B proxy endpoint that returns token
log-probabilities. The runner checks advertised model IDs before execution;
this does not independently attest the server's weight revision.

Qwen CE, Jina CE, SigLIP and CLIP revisions are pinned in the example. Downloads
finish before measured execution starts. AIQA, CSV and ScaleDoc use frozen E5
text embeddings or CLIP text/image embeddings. LOTUS's upstream cascade helper
is vendored with its license and source hash; it does not require installing the
incompatible full LOTUS package. Details and adapter choices are recorded in
[experiment_alignment.md](docs/artifact/experiment_alignment.md).

For ScaleDoc, obtain the upstream source separately:

```bash
git clone https://github.com/Seurgul/ScaleDoc.git external/ScaleDoc
git -C external/ScaleDoc checkout 33d6e815a84f9ac295336d6ece97f66adab5fa51
export SCALEDOC_SOURCE="$PWD/external/ScaleDoc"
cp experiments/models.example.json models.local.json
# Edit models.local.json with the endpoint IDs, URLs and actual served revisions.
```

ScaleDoc imports its encoder, contrastive loss, augmentation and threshold helpers
from this checkout; the runner records its source hashes. Its upstream augmentation
requires a CUDA client. Other selected arms do not need the ScaleDoc checkout.
Keep local model configuration and external sources out of commits.

### Execute each paper section

Run from the repository root after exporting the data paths, expert endpoint and
expert revision described above. Define this helper once:

```bash
paper_run() {
  python experiments/reproduce.py --device cuda:0 \
    --model-revision "$PROBE_ORACLE_REVISION" \
    --model-config models.local.json --scaledoc-source "$SCALEDOC_SOURCE" \
    --output results/paper --resume "$@"
}

# First verify one small, real benchmark query.
paper_run --experiment main --systems PROBE --queries match113 --seeds 0
```

The following commands **execute**, rather than only print, the corresponding
paper grids. Section names and LaTeX labels identify the artifacts independently
of figure numbering.

| Paper section / artifact | Protocol | Execution command |
| --- | --- | --- |
| End-to-end Comparison: `fig:sys-f1`, `fig:sys-latency` | 14 queries, seeds 0–4; 10% budget or target 0.9; all seven systems | `paper_run --experiment main` |
| Quality–Cost Tradeoff: `fig:sys-tradeoff`, `tab:main-comparison` | Budgets 5/10/15/20/25%; target grid 0.1/0.3/0.5/0.7/0.9 | `paper_run --experiment sweep` |
| Sampling Policy: `fig:ablation` (a) | SAS / uniform / Neyman-stratified / top-K; full 5–25% grid; balanced LR; graph, calibration and routing disabled | `paper_run --experiment sampling` |
| Graph Refinement: `fig:ablation` (b) | LR / bias diffusion / full refinement; full 5–25% grid; shared frozen SAS expert labels | `paper_run --experiment refinement` |
| Calibrated Cascade Routing: `fig:alpha-tradeoff` | Platt / GAM / none; alpha 0/0.1/0.3/0.5/0.7/0.9/1; BARGAIN-A/P/R target sweep | `paper_run --experiment calibration` |
| Representation Model: `fig:repr-models` | Qwen CE / Jina CE / SigLIP / CLIP-base; replace both ranking and proxy features | `paper_run --experiment representation` |
| Expert LLM: `fig:mllm-generalization` | Four experts, each paired with its own full no-proxy reference | `paper_run --experiment experts` |
| Time Breakdown: `fig:time-stack` | Seven measured, exclusive stages, with Spark/client overhead reported separately | `paper_run --experiment timing` |
| Less Selective Queries: `tab:lowsel-results` | Cars Q4 / Ecomm Q6 / Movie Q5; budgets 10% and 25%; target 0.9 | `paper_run --experiment low_selectivity` |
| Sampling illustrations: `fig:encoder-rank`, `fig:sas-multi-process` | Ecomm Q9; actual sampling events and encoder order; figure-only batch size 300 | `paper_run --experiment trace` |

```bash
# Preview the complete grid without Spark, model downloads or expert calls.
python experiments/reproduce.py --experiment all --dry-run

# Execute a budget subset of the sampling ablation.
paper_run --experiment sampling --queries match113 --budgets 10 --seeds 0

# Execute just the served Gemma expert and its matching no-proxy reference.
paper_run --experiment experts --experts Gemma4-26B-A4B --queries match113

# Execute one alternative representation.
paper_run --experiment representation --representations CLIP-base --queries match113
```

`--queries`, `--systems`, `--seeds`, `--budgets`, `--alphas`, `--representations`
and `--experts` select existing grid points. They do not change paper workloads.
A subset is partial reproduction. Main quality requires five actual seeds per
configuration; the other grids default to seed 0, and `--seeds` can request further
runs. Full grids can require millions of expert calls; run clients serially under
the same controlled server settings when comparing latency.

The sampling and graph ablations use `floor(0.8 * B)` training labels. Graph arms
reuse the recorded expert acquisition, including its candidate IDs, labels and
hashes. Shared preparation time and each arm's physical request count are stored
separately. With routing disabled, one-class ablations use a constant prediction from the
observed class, with no extra expert calls. `--ablation-single-class oracle` selects
an explicitly different S3 variant; the reporter rejects mixing it with the default
paper protocol. Complete PROBE execution retains S3 fallback.
BARGAIN-P and BARGAIN-R use an oracle allowance of `N` for the current paper's
unlimited-budget comparisons.

Execution support does not certify the paper's numerical claims. Small real-model
checks and protocol tests are recorded in
[validation.md](docs/artifact/validation.md); complete grids, controlled speedup
comparisons and exact paper-layout figure reconstruction remain unverified.
Historical `experiments/01_...sh` through `09_...sh` scripts retain older settings.
The appendix is disabled in the recorded manuscript and is outside these ten grids.

### Outputs and resume

Each job receives a directory derived from its configuration, source hashes,
model bindings, device and data paths:

```text
results/paper/<run-id>/
  completed.json                 # job identity, measured row and output hashes
  attempt-001/
    resolved.json                # resolved job and source hashes
    stdout.log                   # Spark/client execution log
    oracle.jsonl                 # actual expert responses and call stages
    pipeline.json                # policy decisions and measured stage times
    candidate_predictions.npz    # stable IDs and decisions (proxy arms)
    frozen_acquisition.json      # shared expert labels (graph ablation)
    proxy_oracle.jsonl           # actual helper responses (LOTUS)
    encoder_ranking.npz           # ordering scores (trace experiment)
    sampling_trace.json          # actual sampling events (trace experiment)
    *_paper_timing.json          # seven stages and separate client/Spark overhead
    <query>_results.csv           # query metrics and profiling
    *_result.csv/part-*.csv       # raw Spark prediction output
```

Files specific to another arm are absent. `results/paper/frozen/` stores locked,
hashed acquisitions for graph ablations. `--resume` skips intact completed jobs;
interrupted jobs restart in a new preserved attempt and may repeat expert requests.
Completed outputs are not overwritten. `--timeout-seconds` sets a per-job timeout
(default 24 hours). The reporter validates raw hashes, candidate counts, human-GT
provenance and expert failures before aggregating:

```bash
python experiments/report.py --input results/paper --output results/report
```

It records missing configurations explicitly. Trace files contain execution data;
they do not fabricate labels or infer latency from delegation rates.

## License

PROBE retains its [MIT license](LICENSE). Vendored systems, benchmark data and model
weights retain their upstream terms; see [NOTICE.md](NOTICE.md).
