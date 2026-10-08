# PROBE VLDB artifact

The public algorithm defaults follow the active manuscript at `c5b2d86`. Current release status is **algorithm and integration verified, complete paper reproduction pending**. See `algorithm_alignment.md`, `experiment_alignment.md`, `validation.md` and `paper_manifest.json` for distinct implementation and evidence states.

## Installation and environments

```bash
git clone https://github.com/PROBEProject/PROBE.git
cd PROBE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.client.txt
export JAVA_HOME=/path/to/jdk11
export SPARK_LOCAL_IP=127.0.0.1
python experiments/reproduce.py --experiment main --systems PROBE --dry-run
```

The current client pins NumPy1.26.4, SciPy1.17.1, sklearn1.8.0, pandas2.3.1, Spark3.3.1, PyArrow23.0.0, Torch2.9.1 and Transformers5.3.0. Python3.12/Java11 with Arrow is the combination actually tested here; the package pins alone do not imply support for every Spark/Python combination. FAISS is optional; sklearn supplies a CPU neighbor-search fallback. Large candidate populations need substantially more memory and time on that fallback.

`requirements.core.txt` installs the CPU numerical modules. `requirements.client.txt` installs the Spark/CE client. Existing `environment.run*`, `environment.deploy*`, `requirements.run.lock.txt`, `requirements.deploy.lock.txt` and `create_*_env.sh` are historical environment snapshots; they are not certified current-paper environments. Install the vLLM server separately. `validation/environment.json` records observed client/server versions, hardware and provenance scope. The existing server was left running; this task did not switch models or invalidate its caches.

The paper uses two RTX PRO6000 Blackwell Max-Q GPUs, a20-core Xeon w7-3445 and128GB RAM. Full Gemma serving and a separate CE forward need enough GPU memory. The smoke ran alongside an idle, already-loaded tensor-parallel Gemma server; its timings are integration observations, not controlled latency comparisons. Five full runs per main-quality configuration can entail millions of oracle calls; use the printed plan's candidate counts before launching. Per-query latency and download/compute costs remain hardware-dependent and are not yet measured for the new complete grid.

## Data and models

- Obtain SemBench from [the upstream repository](https://github.com/SemBench/SemBench) and use its scenario preparation scripts. The existing vendor snapshot contains loaders and human annotations. Prepared input layout is `PROBE_DATA_ROOT/SemBench/files/<scenario>/data/sf_<factor>/`; human GT belongs under each scenario's `raw_results/ground_truth/`.
- Obtain LRO-Bench from [its upstream repository](https://github.com/LROBench/LROBench). Set `LROBENCH_DATABASES_DIR` to its prepared database tree. Existing `run.py` loaders/scorers define the selected joins. File versions and checksums for the complete released input dataset still need an authoritative manifest; the code does not invent them.
- CE defaults to [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B), revision `4bd860ac4f15ad1897a214615cccc700f8f71818`. Hugging Face downloads the immutable snapshot; content-addressed CE caches include preprocessing, predicate, IDs, source contents and backend identity.
- The default expert is [Gemma4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it). Download/serve under its access terms and record the immutable revision and server configuration. A served model name alone does not establish the weight revision.
- Model changes and alternative representation arms need their own revisions and licenses. Large weights/images/results are not copied into Git. The source license and benchmark/model licenses are separate; see `NOTICE.md`.

Do not derive an experiment from human GT labels or use GT-oracle replay as real expert evidence. Each completed paper run requires human-GT hash provenance, raw prediction output and source hashes. New oracle journals contain candidate IDs, stage, label, response content and token usage; treat them as benchmark output with the data provider's terms.

## Public Spark example

```python
import sem_ops_register
from pyspark.sql import SparkSession
from learned_calibration import paper_calibration_config

spark = SparkSession.builder.master("local[1]").getOrCreate()
df = spark.range(4).selectExpr("id", "'a red car' AS text")
result = df.sem_filter(
    prompt="Does this describe a car? {text}",
    model="google/gemma-4-26B-A4B-it",
    url="http://127.0.0.1:8000/v1/chat/completions",
    embedding_model="unused-in-ce-mode",
    proxy_sample=True,
    learned_calibration_config=paper_calibration_config(2),
)
```

`proxy_sample=True` selects the Python PROBE path. Default SAS, CE representation and calibrated routing are resolved at the public boundary. `B` counts preparation; one-class/empty training samples invoke the full remaining-query S3 fallback. No-proxy uses the same messages and stable-ID oracle transport on the full universe.

## Experiments and outputs

Every active experiment has a named execution path. Append `--dry-run` to preview without downloads, Spark or expert calls:

```bash
python experiments/reproduce.py --experiment main --dry-run
python experiments/reproduce.py --experiment sweep --dry-run
python experiments/reproduce.py --experiment sampling --dry-run
python experiments/reproduce.py --experiment refinement --dry-run
python experiments/reproduce.py --experiment calibration --dry-run
python experiments/reproduce.py --experiment representation --dry-run
python experiments/reproduce.py --experiment experts --dry-run
python experiments/reproduce.py --experiment timing --dry-run
python experiments/reproduce.py --experiment low_selectivity --dry-run
python experiments/reproduce.py --experiment trace --dry-run
python experiments/reproduce.py --experiment all --dry-run
```

Start with an audited executable arm:

```bash
python experiments/reproduce.py --experiment main --systems PROBE \
  --data-root /path/to/prepared-data --model-revision YOUR_IMMUTABLE_EXPERT_REVISION \
  --output results/paper --resume
python experiments/report.py --input results/paper --output results/figures
python experiments/report.py --input results/paper --output results/figures --plot-all
```

Default main plans include all14 queries and seeds0..4. `--queries match113 --seeds 0` selects a partial paper run; it does not become full reproduction. Execution is serial. Each source/config/data-root identity has a run lock and unique output directory. Resume skips intact completed jobs; an interrupted job restarts in a new preserved attempt and may repeat oracle cost. This is job-level recovery, not continuation inside an interrupted oracle batch. No existing run output is overwritten.

Runs reject a candidate count different from paper N. Completion also requires prediction files, GT provenance and zero oracle failures. The executor records source hashes, resolved configuration, raw predictions, oracle journal and profiling. Expert revision is declared by the operator; independently archived server metadata is still required to verify it.

Reporting validates output hashes and protocol, separates query metrics, rejects duplicate seeds, gives Student-t95% CI only for complete five-run main-quality groups, and writes coverage gaps. A geometric mean containing zero is zero; missing queries are not dropped from a complete-paper claim. `--plot-all` refuses incomplete experiment coverage. Any plots it emits are diagnostic quality/delegation views, not a claim that original paper layouts or every numeric conclusion have been reproduced.

All ten families have executors, including frozen-label refinement, alternate representations/experts, seven-stage timing, traces and baseline adapters. Missing endpoint/revision/source prerequisites fail before expert calls. Full grids and exact paper-layout plots remain unverified; historical PDFs/workbooks do not fill those evidence gaps. See `experiment_alignment.md` for explicit adapter choices.

## Validation and troubleshooting

```bash
python -m pytest -q tests
python experiments/smoke.py --output results/new-smoke --device cuda:0
```

Spark tests require the client dependencies and Java11. A Java gateway error usually means `JAVA_HOME` names an unavailable or incompatible runtime. CE mode needs no SigLIP directory; a CE download error should report the requested model/revision. Configure `PROBE_ORACLE_URL`, `PROBE_DATA_ROOT`, CLI output/device/seed, and the expert revision explicitly. Never start duplicate queues or replace another experiment's server.

The artifact separates availability from reproducibility, following [PVLDB's requirements](https://www.vldb.org/pvldb/reproducibility/) for system/environment, input data, experimental scripts and raw-to-figure scripts. Current evidence does not yet support every central paper claim.
