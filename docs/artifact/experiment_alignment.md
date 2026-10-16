# Experiment alignment and evidence gaps

`experiments/paper_manifest.json` defines the active grids, candidate universes,
budgets, seeds and benchmark metrics. `manuscript_snapshot.json` identifies the
manuscript bytes by source hashes. All ten families now execute through `run.py`;
`--dry-run` only previews their configurations. Execution support and completed
numerical reproduction are separate evidence states.

| Active label | `--experiment` | Implemented execution | Remaining evidence |
| --- | --- | --- | --- |
| fig:sys-f1, fig:sys-latency | main | 14 queries, five seeds, seven systems; actual model calls and benchmark scoring | Complete grids and controlled model-matched timing |
| fig:sys-tradeoff, tab:main-comparison | sweep | Budget 5/10/15/20/25%; target 0.1/0.3/0.5/0.7/0.9; BARGAIN uses fixed B10 proxy preparation | Complete measured curves and operating-point tables |
| fig:ablation (sampling) | sampling | SAS/uniform/Neyman/top-K, same CE features and balanced LR; 60% of each preparation budget; no graph/calibration/routing | Complete 14-query budget curves |
| fig:ablation (refinement) | refinement | One locked, hashed acquisition of actual SAS expert labels per input/model/seed/budget; LR/bias/full reuse it | Complete curves; shared acquisition identity checked in each measured group |
| fig:alpha-tradeoff | calibration | Platt/GAM/none, seven alpha values; BARGAIN-A/P/R with separate target points | Full measured grids and human-annotation ECE |
| fig:repr-models | representation | Pinned Qwen/Jina CE and SigLIP/CLIP; ordering and proxy/graph features both change | Complete representation comparisons |
| fig:mllm-generalization | experts | Explicit four model bindings; each PROBE arm has a model-matched no-proxy reference | Three additional expert servers and full four-model grids |
| fig:time-stack | timing | Exclusive wall-clock timers for embedding, sampling, expert calls, training, prediction, refinement and calibration; separate client/Spark overhead | Complete controlled timing panel |
| tab:lowsel-results | low_selectivity | Cars Q7 N30130 (image-only dented predicate), Ecomm Q6 N9280, Movie Q5 N65024, B10/25; all selected baselines | Full measured case-study grids |
| fig:encoder-rank, fig:sas-multi-process | trace | Ecomm Q9 N25440, batch300; exports actual scores/order and sampling events | Complete measured trace |

The runner verifies advertised endpoint model IDs, requires immutable revisions,
excludes weight downloads from timing, isolates client caches and preserves raw
outputs. Served revision declarations still require independently archived server
metadata to attest the actual loaded weights. Single-class training uses a constant prediction in ablations whose routing is
disabled. This under-specified boundary policy is explicit in the manifest and
metadata; `--ablation-single-class oracle` selects a separate S3 variant. Full
PROBE execution retains S3 fallback. Routing and fallback requests
are additional to the PROBE preparation allowance. BARGAIN-P/R use allowance N
for the current manuscript's unlimited-budget comparisons.

## Baseline adapters

Every baseline receives the same candidate IDs, expert messages, expert endpoint
and binary-answer transport. Human GT is used only by the benchmark scorer.

- AIQA uses the existing UQE active sampler: uniform initial batch, logistic
  regression, decaying exploration; batch16, initial exploration1, decay0.9.
  Training purchases B% labels. Its final LR uses the established unbalanced UQE
  policy. One-class training predicts the observed class without buying further
  labels, preserving the sampling-only budget.
- CSV implements equal-vote recursion with k-means4 initially and2 on reclustering,
  B% per-cluster samples and 90% consensus. Adapter choices not specified in the
  manuscript are explicit: at least one sample per nonempty cluster, at most three
  levels, then expert evaluation of unresolved rows. Purchased labels are retained.
- LOTUS filters make actual Qwen3-VL-2B helper calls and archive token log-probabilities.
  Joins use E5/CLIP left-right cosine similarity and the upstream similarity calibration.
  The filter adapter
  calibrates them by quantiles and calls the upstream 1.1.3 importance sampler and
  threshold learner. Those algorithm bodies/defaults are vendored under Apache-2.0
  with exact source provenance. Package imports and logging are adapted. The
  binary convention, calibration replacement sampling, IS weight0.9, quantiles50
  follow that upstream helper. The importance-sampling range covers the entire
  candidate population, overriding the upstream prefix default of200 so its
  correction weights correspond to the sampling distribution. Proxy and expert
  requests are recorded separately. Known calibration labels override proxy
  guesses; no GT population labels enter the threshold learner.
- ScaleDoc imports the actual encoder, loss, augmentation and threshold helpers
  from an explicitly supplied upstream checkout (validated source revision
  `33d6e815a84f9ac295336d6ece97f66adab5fa51`). Training/calibration each purchase
  floor(5% N) expert labels, with disjoint IDs. The adapter uses the upstream
  50/30-epoch phases, Adam parameters, best phase1 checkpoint and final phase2
  weights. Upstream active augmentation duplicates with zero noise. The local
  calibration allocation uses a fixed total and largest remainder over64 bins,
  includes the final edge and retains known labels. These allocation/retention
  adaptations are recorded explicitly. Insufficient training classes or invalid
  training checkpoints invoke a reported full-expert fallback.
- AIQA/CSV/ScaleDoc use frozen E5 for text and CLIP for multimodal data. Native
  Transformers inference avoids the incompatible full LOTUS/SentenceTransformer
  dependency combination. Text uses E5 prefixes and attention-masked mean pooling;
  CLIP uses its pretrained projections. Available text/image and join-side vectors
  are normalized and averaged, with strict image decoding. These feature choices
  are implementation settings where the manuscript does not specify a checkpoint.
- BARGAIN imports the vendored implementation and replaces only PROBE calibration
  and routing, retaining the actual refined scores and shared expert-label cache.

These settings are exposed by source hashes and per-run metadata. They are not a
claim that unspecified historical baseline hyperparameters have been recovered.

## Workload and provenance

The old `match202` entry incorrectly evaluated restaurant matching in zip60642
(270 candidates). It now follows the benchmark's high-rated movie query: IMDb
rating>8.3, Rotten Tomatoes rating>8.5, 24x16=384 candidates, three reference pairs.
See the [official LRO-Bench metadata](https://github.com/LROBench/LROBench/blob/main/match_metadata.json).
Candidate-count checks occur before expert calls, including no-proxy runs.

Main quality requires five actual seeds per exact variant/query. A subset, another
candidate universe, GT replay or borrowed variance cannot substitute for that
coverage. Animals binary-row F1 and Movie semantic-set quality remain
explicit metric identities. Reporting rejects duplicate seeds and corrupt outputs,
retains zero geometric means and reports missing configurations.

The paper's numeric claims remain unverified raw evidence in the manifest. Historical
workbook cells and simulated latency are not measured runs. `paper_statistics.py`
computes paired bootstrap from exported predictions and annotations. Main intervals,
paired differences and geometric-mean groups retain the source-record hashes.
`paper_plots.py` renders all ten families and operating-point tables. Layouts use
new measured data and do not copy values from the paper's original figures.

Calibration ECE uses ten equal-width bins and human annotations outside L and A.
These evaluation choices are explicit because the manuscript does not specify its
binning or evaluation population. Missing calibration fits are reported as undefined,
not averaged away. Retrieval bootstrap units are distinct projected result keys,
matching benchmark set scorers. Physical candidate counts are recorded separately.
Animals Q1 now evaluates binary row-level F1 on 1,600 images, as specified by the paper.
Repeated image paths retain distinct row IDs. The historical count scorer is removed.
Ecomm Q8 retains its image–text query, although the workload table calls it text-only.

Input contents are hashed outside timing before any resume decision. Comparisons
require shared source, input, expert and GT identities, and refinement arms must
share acquired labels. Changing a plotting script does not invalidate execution.
The original SemBench vendor revision and independently attested live-server weights
remain unavailable. The appendix is disabled. See `validation.md` for the exact
scope of tests and real-model integration evidence.
