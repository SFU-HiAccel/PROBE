# LASO: A Proxy-Based Approximate Semantic Query Engine


LASO is a proxy-based approximate query processing (AQP) engine for **semantic SQL
operators** (`sem_filter`, `sem_join`). AI query systems powered by large language
models (LLMs) let operators such as filters and joins evaluate complex natural-language
predicates over unstructured, multimodal data, but scaling them is expensive: every
input tuple costs one or more LLM invocations. A semantic join over a 115K-row table,
for example, takes more than 11 hours of LLM calls.

LASO reduces this cost by training a lightweight **proxy classifier** on a small sample
labeled by an expensive expert MLLM and using the proxy to evaluate the remaining rows.
The central difficulty is **high selectivity**: when matching rows are extremely rare
under the benchmark's human reference labels, naive sampling fails to collect enough
informative examples to fit an accurate proxy, and a single-shot proxy generalizes
poorly. LASO addresses this with three ideas that share a single 1-D ordering axis built
from embedding similarity:

1. **Active-learning sampling** concentrates the LLM budget on rows near the predicate's
   decision boundary, where labels are most informative.
2. **Graph-based label propagation** uses the unlabeled rows to refine the proxy's
   predictions at no extra LLM cost.
3. **An online model router** decides, per row, whether to trust the proxy or fall back
   to the expert MLLM.

On **SemBench** and **LRO-Bench**, LASO matches the mean F₁ of the SparkSQL AI UDF
(no-opt) baseline while delivering up to **212× speedup** (9.5× geometric mean) with a
**16.34× reduction in token usage**.

> LASO is the only system among Lotus, ThalamusDB, and UQE that preserves the no-proxy
> baseline F₁ while also cutting cost.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│  PySpark DataFrame  (input table, or candidate pairs from a join)      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │   df.sem_filter(...) / df.sem_join(...)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  sem_ops_register.py — registers semantic operators on Spark DataFrames│
│  Routes to the proxy pipeline when proxy_sample=True                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  proxy_model.py — Multimodal embedding (SigLIP, 768-d) + cosine        │
│  similarity against the query predicate → the shared 1-D axis σ         │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cluster_proxy.py — the three-stage LASO pipeline                       │
│                                                                         │
│   ① Active-learning sampling    select B boundary rows, label via MLLM  │
│   ② Logistic-regression proxy    base score z for every candidate       │
│   ③ Label propagation            bias diffusion + homogeneity prop.     │
│   ④ Online model router          per-row: trust proxy or recheck w/ LLM │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Filtered / joined PySpark DataFrame                                    │
└──────────────────────────────────────────────────────────────────────┘
```

All three stages reuse the same shared 1-D axis σ derived from SigLIP cosine similarity
between each row's embedding and the query embedding. For joins, each candidate pair
`(left, right)` is embedded by concatenating and mean-pooling the two side embeddings;
candidates come from the host engine's blocking step, so LASO scores only the pairs the
planner materialized rather than the full cross product.

---

## The Three Components

### 1. Active-Learning Sampling — `cluster_proxy.py`

Selecting which `B` of `N` rows to send to the expert MLLM is cast as a **one-dimensional
threshold search** on the ordering axis. The axis assumes a single threshold `t*`
separates matches from non-matches; under high selectivity the matching rows cluster at
high similarity above this threshold.

The sampler runs for `T` rounds. Round `t`:

1. Builds the ordering axis `aᵢ = (1-λ_t)·z(sᵢ) + λ_t·z(ℓᵢ)`, a z-score blend of the
   embedding similarity `sᵢ` and the current logistic-regression proxy score `ℓᵢ`. The
   proxy weight `λ_t` ramps linearly from `0` (cold start) in round 1 to `1` in round `T`.
2. **`LocateThreshold`** spends an *explore* fraction `ρB` of the budget contracting a
   candidate interval `[I, J]` toward the separating threshold: each iteration samples a
   small batch, queries the LLM, recomputes the threshold loss `L̂(s)` (labeled
   non-matches above `s` plus labeled matches below `s`), and geometrically shrinks the
   interval by a factor `c` to the sub-interval whose endpoints have the lowest loss.
3. Spends the remaining budget `(1-ρ)B` labeling unlabeled rows **on the match side**
   (right of `t*`) first, walking outward from the boundary, then uniformly on the
   non-match side if the match side is exhausted.

Across rounds the axis shifts from pure similarity toward the learned proxy score and the
threshold sharpens, so the labeled set concentrates near the boundary — collecting enough
matching rows to fit an accurate proxy even when matches are a fraction of a percent.

### 2. Graph-Based Label Propagation — `cluster_proxy.py`

A logistic-regression proxy trained on the small labeled set scores every candidate
(`Z`), but generalizes imperfectly. Label propagation refines `Z` on a **k-NN graph** over
the row embeddings, exploiting two properties: **error correlation** (the proxy's errors
are correlated across graph neighbors) and **homophily** (adjacent rows tend to share a
label). The propagation operator is `S = D^{-1/2} W D^{-1/2}`, where
`Wᵢⱼ = exp(-‖eᵢ-eⱼ‖² / 2σ²)` is a Gaussian edge weight and `σ` is auto-tuned to the median
k-NN distance.

- **Phase A — Bias diffusion** measures the residual error `yᵢ - Zᵢ` on labeled rows and
  diffuses it along the graph (damping `α`), adding the diffused residual back to `Z` to
  repair locally biased predictions → adjusted scores `z^adj`.
- **Phase B — Homogeneity propagation** pins labeled rows at their MLLM labels, initializes
  the rest with `z^adj`, and diffuses again (damping `β`) so neighboring rows agree →
  propagated scores `z^prop`.

The proxy thresholds `z^prop` at `θ` to assign every candidate's label `ŷᵢ = 𝟙[z^prop ≥ θ]`.

### 3. Online Model Router — `learned_deferral.py`, `cluster_proxy.py`

A proxy that is accurate on average can still be wrong on specific rows. Instead of a
threshold on `z^prop`, LASO trains a **per-row router** that estimates the probability the
proxy's label disagrees with the MLLM, and escalates the highest-risk rows under a
verification budget `B_v`.

The active-sampled labels are stratified by MLLM label and split into `B_proxy` (trains the
proxy) and `B_defer` (held out). A logistic regression `g(x) ≈ Pr[proxy disagrees with LLM]`
is fit on `B_defer` using three zero-cost features that are by-products of the upstream
stages:

- `z^prop` — the propagated proxy score,
- `nd` — a neighbor-disagreement score, the Gaussian-weighted average gap between a row's
  `z^prop` and its graph neighbors' (large `nd` ⇒ the row straddles the boundary),
- `ρ` — the row's rank on the shared similarity axis, rescaled to `[0,1]`.

At query time the router accepts the proxy on rows it predicts will agree, and routes the
top-`B_v` highest-risk rows to a fresh LLM call.

---

## Repository Layout

| File | Purpose |
|------|---------|
| `proxy_model.py` | SigLIP multimodal embedding, expert-MLLM labeling, `sem_filter` / `sem_join` proxy entry points |
| `cluster_proxy.py` | The three-stage pipeline: active-learning sampling, k-NN graph, label propagation, online model router |
| `sem_ops_register.py` | Registers semantic operators (`sem_filter`, `sem_join`, `sem_map`, `sem_topk`, …) on PySpark DataFrames |
| `learned_deferral.py` | Online model router: feature construction, training, per-row fall-back decision |
| `cascade_split.py` | Stratified `B_proxy` / `B_defer` split feeding the router |
| `cs_calibration.py` | Dual-threshold calibration for the propagated (Correct & Smooth) scores |
| `proxy_model_backends.py` | Pluggable embedding backends (SigLIP, CLIP, Jina, GME, …) |
| `run.py` | **Main benchmark harness**: ablation grid over queries, sampling methods, and budgets |
| `run_lrobench_select127_grid.py` | LRO-Bench low-selectivity grid (mirrors the main harness) |
| `run_configured_runs_xlsx.py` | Pre-configured batch runner that consolidates results into a spreadsheet |
| `sembench_q5_proxy_model.py`, `sembench_q6_proxy_model.py`, `sembench_q7_proxy_model.py` | Per-query `sem_join` experiment drivers |
| `cars_Q1.py`, `cars_Q3.py`, `cars_Q4.py` | Standalone cars-scenario query scripts |
| `viz_similarity_distribution.py`, `viz_stacked_distribution.py` | Separability analysis: similarity histograms, ROC-AUC |
| `pareto_frontier.py` | Cost–quality Pareto-frontier figures from the per-query CSVs |
| `delegation_curves.py` | Router / delegation-curve figures from the per-query CSVs |
| `dataset/*_groundtruth.py` | Pairwise-join ground-truth generators (company, flickr, roxford, veri) |
| `tests/` | CPU-only smoke and unit tests for the pipeline components |
| `results/similarity_dist/` | Example separability figures and `separability_summary.csv` |

---

## Workload

LASO is evaluated on 14 highly selective filter (`F`) and join (`J`) queries spanning text
and multimodal predicates, drawn from **SemBench** and **LRO-Bench**:

| Query | Scenario | Type | Modality | Candidates | Matching Rate |
|-------|----------|:----:|----------|-----------:|--------------:|
| mmqa_q2a   | MMQA       | J | text+image | 10,400  | 5.0e-4 |
| mmqa_q7    | MMQA       | J | text+image | 230,400 | 2.6e-5 |
| ecomm_q7   | E-commerce | J | text       | 525,625 | 9.4e-3 |
| ecomm_q8   | E-commerce | J | text       | 115,768 | 3.0e-4 |
| ecomm_q9   | E-commerce | J | text+image | 25,440  | 2.65e-2 |
| ecomm_q13  | E-commerce | F | text+image | 3,992   | 3.0e-3 |
| cars_q3    | Cars       | F | text+image | 10,070  | 3.39e-2 |
| cars_q4    | Cars       | F | text       | 157,269 | 1.39e-1 |
| cars_q8    | Cars       | F | text+image | 30,130  | 5.4e-3 |
| animals_q1 | Animals    | F | text+image | 1,600   | 6.1e-2 |
| LRO_match113 | santos   | J | text       | 266     | 6.67e-3 |
| LRO_match121 | santos   | J | text       | 1,426   | 4.91e-3 |
| LRO_match202 | movies   | J | text       | 384     | 7.81e-3 |
| LRO_match305 | schools  | J | text       | 1,421   | 7.74e-3 |

Quality is reported as F₁, precision, and recall against SemBench / LRO-Bench human
reference labels. The human labels are used only for evaluation — LASO observes only the
labels returned by the expert MLLM during sampling and rollback.

---


### Hardware & software environment

- **OS & Software:** Linux (tested on Ubuntu). **Python:** 3.10–3.12. **Java:** 8 or 11 (for PySpark 3.3).  CUDA 12.4 toolkit, vLLM 0.20.0, and transformers 4.57.0
- **GPU** Two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs
- **CPU** Intel Xeon w7-3445 with 20 cores, with 125 GB of RAM. 
- An **OpenAI-compatible LLM endpoint** (any server works; we use vLLM). Default endpoint
  `http://localhost:8000/v1`.

### Installation

```bash
# Use this to create the deploy conda env
bash /LASO/create_deploy_env.sh

# get into the deploy conda env
conda activate deoloy

export HF_HOME=$HOME/hf-cache
export VLLM_CACHE_DIR=$HOME/vllm-cache
export TORCH_COMPILE_CACHE=$VLLM_CACHE_DIR/torch_compile_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=$HOME/vllm-compile-cache
export TORCHINDUCTOR_CACHE_DIR=$HOME/vllm-compile-cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export LD_LIBRARY_PATH="$HOME/.local/lib/:$LD_LIBRARY_PATH"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH

export CUDA_INCLUDE_DIR=$CONDA_PREFIX/targets/x86_64-linux/include
export NVCC_PREPEND_FLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_ROOT=$CONDA_PREFIX
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CMAKE_ARGS="-Dprotobuf_BUILD_TESTS=OFF"
export CFLAGS="$CFLAGS -Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"

vllm serve "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --enable-mm-embeds \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --port 8000 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --allowed-local-media-path /local-scratch/localhome/hza214/SemBench \
  > vllm_server_30b.log 2>&1 &

```

The run scripts call `http://localhost:8000/v1/chat/completions` and snapshot vLLM's
Prometheus `/metrics` to count tokens for the (Java-side) baseline path. Override the
endpoint with `VLLM_METRICS_URL` if you serve elsewhere.




### Data and models

#### Download links

Both benchmarks are **public, third-party artifacts** (not authored by us); LASO
consumes them unmodified. Obtain each from its canonical source below, then point
`$LASO_DATA_ROOT` at the result (layout in the next subsection).

| Benchmark | What LASO uses | Code / paper | Data download |
|-----------|----------------|--------------|---------------|
| **SemBench** | cars / ecomm / movie / mmqa / animals scenarios | [github.com/SemBench/SemBench](https://github.com/SemBench/SemBench/) · [arXiv:2511.01716](https://arxiv.org/abs/2511.01716) | Auto-downloaded from Google Drive on first harness run (per-scenario file IDs below) |
| **LRO-Bench** | `santos` tables for the four `LRO_match*` join queries | [github.com/LROBench/LROBench](https://github.com/LROBench/LROBench/) · [arXiv:2603.02537](https://arxiv.org/abs/2603.02537) | Clone the repo's database CSVs (27 DBs from BIRD / Magellan / NextiaJD / Santos) |

**SemBench** ships an automatic downloader (vendored at `third_party/sembench`): its
harness pulls each scenario's multimodal archive from Google Drive on first run and
materializes the tables locally. The per-scenario Google Drive file IDs wired into the
downloader (direct link form: `https://drive.google.com/file/d/<file_id>/view`) are:

| Scenario | Google Drive file ID |
|----------|----------------------|
| cars     | `1ER5pooCIi2q6ZTYUrJw_ha_X3byNQB9M` |
| ecomm    | `1fVV9PLgIMT-e-zxFM5ksdnN8xQlfgnmb` |
| movie    | `1WkeXN3V5A6h-D_oeWvv2G_xs2PgzcQu0` |
| mmqa     | `1oHXq5oxIfsyoNy9aCQ0V9G3W4puqHC6i` |
| animals  | `1HG6tvXIA0BtpqbZqCR46oNSeY2yZ4Lko` |

**LRO-Bench** distributes its relational databases through the LROBench repository above.
LASO's four join queries (`LRO_match{113,121,202,305}`) read only the `santos` tables;
clone the repo and place its `databases/` tree where `$LROBENCH_DATABASES_DIR` points
(default `$LASO_DATA_ROOT/databases/`).

> **Licenses.** SemBench and LRO-Bench retain their upstream licenses (see each
> repository); the underlying media inherit their original dataset licenses (e.g. the
> SemBench scenario sources on Kaggle / Google Drive, and the public-sector `santos`
> tables). LASO's own code is MIT.

#### Directory layout

All data/model locations are resolved from the environment variable
`LASO_DATA_ROOT` (default: your home directory). No paths are hardcoded. Lay out the
inputs as follows and point the variable at the root:

```bash
export LASO_DATA_ROOT=/path/to/data_root
```


- **SemBench** provides the cars / ecomm / movie / mmqa / animals scenarios.
- **LRO-Bench** provides the `LRO_match*` join queries; `run_lrobench_select127_grid.py`
  reads its tables from `$LASO_DATA_ROOT` as well.
- **Pairwise-join ground truth** for the company/flickr/roxford/veri scenarios can be
  regenerated with the `dataset/*_groundtruth.py` scripts (each takes `--data_dir`).
- **Embedding model:** any SigLIP-class encoder; the default is `siglip-base-patch16-224`.
  Override with the `EMBEDDING_MODEL` constant at the top of the run scripts.

> If your directory layout differs, either symlink it under `$LASO_DATA_ROOT` with the
> structure above, or edit the per-scenario `data_dir` / `gt_path` defaults near the top
> of `run.py`.

#### Downloading the benchmark data

Download each benchmark from its canonical source (full link table in "Download links"
above), then materialize it with its own tooling (vendored under `third_party/`, see
"Reproducing the Baselines" below) and point `$LASO_DATA_ROOT` at the result.

**SemBench** — [github.com/SemBench/SemBench](https://github.com/SemBench/SemBench/)
([arXiv:2511.01716](https://arxiv.org/abs/2511.01716)). The cars / ecomm / movie / mmqa /
animals scenarios **auto-download from Google Drive and build** their multimodal databases
on the first run of the harness (per-scenario Google Drive file IDs listed under "Download
links" above) — no manual download step:

```bash
cd third_party/sembench
pip install -r requirements.txt            # into the `sembench` env
# Downloads + materializes data into third_party/sembench/files/<scenario>/data/sf_<sf>/
python3 src/run.py --systems lotus --use-cases cars ecomm mmqa animals movie \
        --scale-factor <sf>
# Re-use the materialized data on later runs (skip re-download/build):
python3 src/run.py --systems lotus --use-cases cars --queries 3 --skip-setup ...
```

Each scenario has its own scale-factor range; pick the `--scale-factor` whose
candidate-set size matches the workload table above (e.g. the full `cars` table is
~157K rows, `mmqa_q7` ~230K). The downloaded media live under
`third_party/sembench/files/*/data/` and are git-ignored — only query definitions
and `raw_results/ground_truth/` human labels are vendored.

**LRO-Bench** (the `LRO_match{113,121,202,305}` join queries) is built from the
LROBench / BIRD relational databases. Clone the database CSVs from the public LROBench
repository ([github.com/LROBench/LROBench](https://github.com/LROBench/LROBench/),
[arXiv:2603.02537](https://arxiv.org/abs/2603.02537)) and place them under
`$LASO_DATA_ROOT/databases/` (the tree BARGAIN's grid reads via the
`LROBENCH_DATABASES_DIR` environment variable, and LASO's harness reads for the
`LRO_match*` queries). The four paper queries draw from the `santos`,
`restaurants2`, and schools tables.

### Run the experiments

Each paper experiment has a dedicated script under **`experiments/`** that configures the
`run.py` harness for exactly that experiment (full map in `experiments/README.md`). This is
the recommended entry point.

**Claim → command map.** Every paper table/figure maps to exactly one script:

| Paper artifact | Claim it supports | Command |
|----------------|-------------------|---------|
| Table 2, Fig 1–2 | Quality parity + speedup vs. baselines | `bash experiments/01_main_system_comparison.sh` |
| Table 3 | Sampling-policy ablation | `bash experiments/02_sampling_policy_ablation.sh` |
| Table 4 | Label-propagation ablation | `bash experiments/03_label_propagation_ablation.sh` |
| Table 5 | Online-router ablation (off vs. on) | `bash experiments/04_online_router_ablation.sh` |
| Fig 3 | Hyperparameter sensitivity (k, ρ, budget) | `bash experiments/05_hyperparameter_sensitivity.sh` |
| Table 6 | Embedding-model robustness (5 embedders) | `bash experiments/06_embedding_model_robustness.sh` |
| Fig 4 | MLLM generalization (4 expert MLLMs) | `bash experiments/07_mllm_generalization.sh` |
| Fig 5–6 | Per-stage time breakdown | `bash experiments/08_time_breakdown.sh` |
| Table 7, Fig 7 | Low-selectivity case study (budget B ∈ {1,5,10}%) | `bash experiments/09_case_study_low_selectivity.sh` |

Run them directly:

```bash
export LASO_DATA_ROOT=

# Use this to create the run conda env
bash /LASO/create_run_env.sh

# get into the deploy conda env
conda activate run

bash experiments/01_main_system_comparison.sh      # Table 2, Fig 1–2
bash experiments/02_sampling_policy_ablation.sh     # Table 3
bash experiments/05_hyperparameter_sensitivity.sh   # Fig 3
# … 9 scripts total, one per experiment (01–09)

# Preview a script's resolved config without launching Spark/LLM:
LASO_DRYRUN=1 bash experiments/01_main_system_comparison.sh
```

`run.py` is the underlying harness. It takes an optional positional argument selecting the
benchmark (or the `LASO_QUERIES` env var); with no argument it runs the full grid. Results
go to **`./results/direct_grid/`** (or `LASO_RESULTS_DIR`).

```bash
# Direct harness invocation (the experiments/ scripts wrap these)
python3 run.py                # full grid
python3 run.py ecomm          # one scenario
python3 run.py ecomm_q7       # a single query
```

Each query prints its selectivity (`gt_count / total`) on first run and writes
`results/direct_grid/<query>_results.csv` (per-row predictions + metrics) and
`results/direct_grid/<scenario>_grid_summary.csv` (F₁, latency, LLM-call count per
configuration). To consolidate the headline numbers into a single spreadsheet:


### Generate the figures and tables

Every figure and LaTeX table in the paper is regenerated by a small script under
**`paper_figures/`**. These read the aggregated results workbook
`paper_figures/plotdata.xlsx` (one sheet per result group) and write PDFs/PNGs to
`paper_figures/figures/` and `.tex` tabulars to `paper_figures/tables/`. They need
only `pandas`, `openpyxl`, `numpy`, `matplotlib` (no GPU, no LLM) — the `run` env
or the lighter `sembench` env both work.

Regenerate everything in one command (from the repo root):

```bash
bash paper_figures/make_figures.sh
```

…or run an individual script. Each script carries a top-of-file docstring note
describing exactly what it draws and which sheet it reads:

| Script | Paper artifact (label) | What it does | `plotdata.xlsx` sheet(s) |
|--------|------------------------|--------------|--------------------------|
| `paper_figures/plot_system_f1.py` | `fig_sys_f1.pdf` (`fig:sys-f1`) | Per-query **F₁** grouped bars across the 7 systems (Lotus, ThalamusDB, UQE, BARGAIN, Baseline, LASO−router, LASO+router) | `ablation-f1-system` |
| `paper_figures/plot_system_latency.py` | `fig_sys_latency.pdf` (`fig:sys-latency`) | Per-query **latency** (log scale), same 7 systems | `ablation-latency-system` |
| `paper_figures/plot_time_breakdown.py` | `fig_time_abs.pdf` (`fig:time-stack`) | **Absolute** stacked-bar time breakdown per query (embedding / sampling / labeling / propagation / proxy-train / proxy-predict / LLM-rollback) vs the full-MLLM baseline | `time-breakdown`, `ablation-latency-system` |
| `paper_figures/plot_hparam_sensitivity.py` | `fig_hparam.pdf` (`fig:hparam`) | 5×3 dual-axis grid: F₁ & latency vs sampling rate, kNN degree *k*, and explore rate ρ | `hyper-param` |
| `paper_figures/plot_model_generalization.py` | `fig_model_generalization.pdf` (`fig:mllm-generalization`) | Per-oracle mean F₁ + geo-mean latency speedup across 4 MLLMs (Qwen3-VL, Qwen3.6, InternVL3.5, GLM-4.5V) | `Model-ablation` |
| `paper_figures/plot_tables.py` | `tables/table3.tex` + `table4.tex` (`tab:sampling-sampler`, `tab:sampling-prop`) | Full LaTeX tabulars for the sampler-comparison and label-propagation ablations (with std-devs) — both written in one run | `Table3-sampler`, `Table4-prop`, `Table34-stddev` |

`paper_figures/plot_style.py` is the shared palette/styling helper (not a figure
generator). The hand-drawn schematic figures in the paper (workflow, label-propagation
pipeline, threshold contraction, teaser) have no generator script — they are authored
diagrams.

### Notes & caveats

- `sembench_q{5,6,7}_proxy_model.py` import the operators as the `omnimcp` package
  (`omnimcp.server.*`); the same modules are vendored at the repo root
  (`proxy_model.py`, `sem_ops_register.py`). Install that package or adjust the imports to
  the local modules to run those per-query drivers.
- `cars_Q3.py` additionally requires the third-party `lotus` library.
- See `CHANGELOG.md` for the most recent changes to the experiment harness.

---


## Reproducing the Baselines

The paper compares LASO against four systems. **Lotus** and **ThalamusDB** are run
through the vendored SemBench copy (`third_party/sembench`), **BARGAIN** through the
vendored BARGAIN copy (`third_party/bargain`), and **UQE** is the in-repo `run.py`
sampler `uqe_active`. All four use the *same* serving stack as LASO: vLLM serving the
expert `Qwen/Qwen3-VL-30B-A3B-Instruct` (and, for the proxy cascades,
`Qwen/Qwen3-VL-2B-Instruct`) over an OpenAI-compatible endpoint (default
`http://localhost:8000/v1`; see §R5). SigLIP is the shared embedder.

### Lotus and ThalamusDB (`third_party/sembench`)

Both run via SemBench's harness, but **take different `--model` strings**, so run them
as separate commands. The Lotus runner adds the LiteLLM `hosted_vllm/` prefix itself
and calls `http://localhost:8000/v1`; ThalamusDB selects a model config by the full
`hosted_vllm/Qwen/...` key (config shipped at
`config/system/thalamusdb/Qwen3-VL-30B-A3B-Instruct.json`).

```bash
cd third_party/sembench
conda activate run
export LASO_DATA_PATH=/localhome/hza214/SemBench/files


# Lotus — pass the BARE model name (runner prepends hosted_vllm/)
python3 src/run.py --systems lotus \
        --use-cases ecomm --queries 7 \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --scale-factor <sf>

# ThalamusDB — pass the hosted_vllm/-PREFIXED model name
python3 src/run.py --systems thalamusdb \
        --use-cases ecomm --queries 7 \
        --model hosted_vllm/Qwen/Qwen3-VL-30B-A3B-Instruct \
        --scale-factor <sf>
```

### BARGAIN (`third_party/bargain`)

BARGAIN cascades the `Qwen/Qwen3-VL-2B-Instruct` proxy before the
`Qwen/Qwen3-VL-30B-A3B-Instruct` oracle (both are the runner defaults). Point it at the
benchmark data via the environment, then run a tag or group:

```bash
cd third_party/bargain
pip install -e .                                   # or: pip install -r ../sembench/requirements.txt
export LASO_DATA_ROOT=/path/to/data_root
export LROBENCH_DATABASES_DIR=/localhome/hza214/LLMSQL/databases
python examples/run_bargain_galaxy_grid.py all \
  --proxy-model  "Qwen/Qwen3-VL-2B-Instruct"     --proxy-url  http://localhost:8001/v1 \
  --oracle-model "Qwen/Qwen3-VL-30B-A3B-Instruct" --oracle-url http://localhost:8000/v1
```

Serve the proxy and oracle on separate endpoints with `--proxy-url` / `--oracle-url`
(or `BARGAIN_PROXY_URL` / `BARGAIN_ORACLE_URL`) if they run on different ports.
Per-query F1 / oracle-usage CSVs are written under `--results-dir`.

### UQE (`run.py`, `uqe_active`)

UQE is the in-repo logistic-regression proxy with active-learning sampling and no
label propagation or router. Select it through the `LASO_ABLATIONS` env var (label
`uqe_active+lr_only`, matching the paper's UQE row); the expert model comes from
`SEMBENCH_MODEL` (default `Qwen/Qwen3-VL-30B-A3B-Instruct`):

```bash
# from the repo root
export LASO_DATA_ROOT=/path/to/data_root
LASO_ABLATIONS="uqe_active+lr_only" python3 run.py ecomm_q7      # one query
LASO_ABLATIONS="uqe_active+lr_only" python3 run.py ecomm         # a scenario
```

The `uqe_active` sampler is implemented in `cluster_proxy.py` (selected via the
`uqe_active` strategy in `run.py`'s `ABLATION_CONFIGS`). The matching
budgets/knn/explore-rate knobs (`LASO_BUDGET_PCTS`, `LASO_KNN_K`, `LASO_VS_RATIO`) are
documented in §R6–R7.

---

## License

Released under the MIT License — see `LICENSE`.
</content>
