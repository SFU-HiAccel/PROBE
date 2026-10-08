# Validation record

Validation on 2026-10-08 used an isolated release worktree. Original dirty research files and the pre-existing Gemma server were preserved. Detailed machine-readable records are under `validation/`.

| Check | Actual evidence | Limits |
| --- | --- | --- |
| Clean numerical environment | Fresh Python3.12 venv, `pip install -r requirements.core.txt`;174 selected CPU tests passed | Numerical/sampling contracts only |
| Clean client installation | Fresh venv then `pip install -r requirements.client.txt` succeeded; exact installed closure saved in `requirements.client.lock.txt` | Separate existing Java11 required; no new vLLM server installation |
| Client regression suite |356 tests passed in the fresh client environment, including oracle journaling and consistent public seed defaults | Model/network doubles in contract tests; final clean-clone rerun recorded separately |
| Dry-run | All10 named experiment plans validated with isolated stdlib Python; no Spark/oracle startup | All ten families have executors; dry-run is configuration evidence only |
| Real public Spark smoke | Qwen CE and Gemma4, text filter/join,4 candidates each;8 physical requests,0 failures | B2 per operation, L1/fallback3 verifies S3; synthetic input, no quality/speedup claim |
| Real benchmark through unified entry | LRO113, actual/paper N266, seed0, B10%: L20/A6/D80,106 unique labels,0 failures, observed GT F1=1.0 | One query/seed, declared expert revision only, no matched no-proxy/control timing comparison |
| Raw validation/report | One completed real run passed hashes, universe, GT hash and zero-failure checks; remaining groups reported missing | No full-grid figures certified |
| Publication scan | Selected tree had no matches for checked token/private-key patterns or personal absolute paths;725 reachable blobs scanned;18 historical blobs had personal paths | Pattern scan is not exhaustive; no history rewrite |

`real_spark_smoke.json` records actual output rows, counters and source hashes. `oracle_smoke.jsonl` contains genuine expert answers on the synthetic fixtures. `measured_run.json` is a path-redacted summary of the real LRO run; the full immutable run and raw files remain in the release worktree's ignored results directory. Source hashes identify the exact executed bytes even when they precede the final packaging commit.

Environment metadata separates the observed client, installed server environment, and un-attested live server revisions. The installed server packages may differ from modules loaded by a server started days earlier; this record does not assert they match. Server prefix cache was enabled and uncontrolled, so observed seconds are not a publishable speedup comparison.

Paper-wide numeric conclusions, five-run coverage across14 queries, additional expert/proxy servers, exact paper-layout plots and the complete public input checksum manifest remain missing or partial. `experiment_alignment.md` and `paper_manifest.json` specify those gaps. Current validation supports the implemented contracts and small integrations, not a completed VLDB reproducibility evaluation.

## Current paper execution update

The runner now implements all ten active experiment families. The regression suite
passed 386 tests; subsequent focused boundary/BARGAIN tests passed 8 cases. The client
now includes torchvision, SentencePiece and protobuf, and `pip check` reports no
broken requirements. All 2,116 resolved configurations are unique and have executors.
`execution_tests.json` records these checks and their limits.

New real-model validation uses the existing Gemma service, prepared LRO inputs and
actual Qwen/Jina/SigLIP/CLIP inference. It checks the corrected 384-candidate movie
query, four sampling policies, three frozen-label graph variants, Platt/GAM/none,
Gemma's matched no-proxy reference, seven-stage timing and baseline integration.
The graph variants' acquisition files have identical hashes; only the first arm
purchases 30 labels, and the next two make zero expert requests.

The one-class boundary was corrected during validation: sampling-only ablations
and AIQA use a constant prediction, preserving their label budget. Earlier runs
that used whole-query fallback are retained as diagnostics. The revised default
is explicit in the protocol; the reporter will not mix the earlier protocol with
current runs. Full PROBE continues to use S3 fallback. A separate explicit S3
ablation variant can be selected, but is not the default paper grid.

ScaleDoc's imported encoder/loss and 50/30-epoch CUDA training were exercised with
synthetic purchased-label fixtures; its calibration adapter retained original IDs
and its exact calibration quota. These are implementation contracts, not real
paper quality evidence. LOTUS's actual token-logprob adapter and unmodified upstream
threshold bodies are covered by unit fixtures; its real Qwen3-VL-2B server is still
needed for measured baseline results. Three other expert servers were not started.

`execution_runs.json` separates current-protocol records from earlier diagnostics,
with source-set and raw-record hashes. Raw attempts stay in ignored local results;
these redacted summaries do not replace them. No full-grid number, confidence
interval or controlled speedup is claimed from these checks.

The final five current-protocol runs passed report validation. Uniform sampling
used 30 labels for 384 candidates and AIQA used 26 labels for 266 candidates, both
without whole-query fallback. The report marked all ten paper grids as partial.
The final publication pattern scan examined 645 selected files and found no
checked token, private-key or personal-path patterns; this is not an exhaustive
secret scan.
