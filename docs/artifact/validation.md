# Validation record

Validation on 2026-10-08 used an isolated release worktree. Original dirty research files and the pre-existing Gemma server were preserved. Detailed machine-readable records are under `validation/`.

| Check | Actual evidence | Limits |
| --- | --- | --- |
| Clean numerical environment | Fresh Python3.12 venv, `pip install -r requirements.core.txt`;174 selected CPU tests passed | Numerical/sampling contracts only |
| Clean client installation | Fresh venv then `pip install -r requirements.client.txt` succeeded; exact installed closure saved in `requirements.client.lock.txt` | Separate existing Java11 required; no new vLLM server installation |
| Client regression suite |356 tests passed in the fresh client environment, including oracle journaling and consistent public seed defaults | Model/network doubles in contract tests; final clean-clone rerun recorded separately |
| Dry-run | All10 named experiment plans validated with isolated stdlib Python; no Spark/oracle startup | Unsupported arms remain plans and stop before expensive execution |
| Real public Spark smoke | Qwen CE and Gemma4, text filter/join,4 candidates each;8 physical requests,0 failures | B2 per operation, L1/fallback3 verifies S3; synthetic input, no quality/speedup claim |
| Real benchmark through unified entry | LRO113, actual/paper N266, seed0, B10%: L20/A6/D80,106 unique labels,0 failures, observed GT F1=1.0 | One query/seed, declared expert revision only, no matched no-proxy/control timing comparison |
| Raw validation/report | One completed real run passed hashes, universe, GT hash and zero-failure checks; remaining groups reported missing | No full-grid figures certified |
| Publication scan | Selected tree had no matches for checked token/private-key patterns or personal absolute paths;725 reachable blobs scanned;18 historical blobs had personal paths | Pattern scan is not exhaustive; no history rewrite |

`real_spark_smoke.json` records actual output rows, counters and source hashes. `oracle_smoke.jsonl` contains genuine expert answers on the synthetic fixtures. `measured_run.json` is a path-redacted summary of the real LRO run; the full immutable run and raw files remain in the release worktree's ignored results directory. Source hashes identify the exact executed bytes even when they precede the final packaging commit.

Environment metadata separates the observed client, installed server environment, and un-attested live server revisions. The installed server packages may differ from modules loaded by a server started days earlier; this record does not assert they match. Server prefix cache was enabled and uncontrolled, so observed seconds are not a publishable speedup comparison.

Paper-wide numeric conclusions, five-run coverage across14 queries, all baseline arms, seven-stage timing, exact paper-layout plots and the complete public input checksum manifest remain missing or partial. `experiment_alignment.md` and `paper_manifest.json` specify those gaps. Current validation supports the implemented contracts and small integrations, not a completed VLDB reproducibility evaluation.
