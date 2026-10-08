# Attribution and release scope

PROBE's license is in [LICENSE](LICENSE). Its existing copyright notice is preserved.

- `third_party/lotus` contains the Apache-2.0 cascade helper from LOTUS 1.1.3, with upstream algorithm bodies and defaults preserved. Package imports and logging are adapted to avoid its incompatible optional dependency stack; `provenance.json` records the exact source hash.
- `third_party/bargain` contains BARGAIN by Sepanta Zeighami and contributors. Preserve its MIT license and source attribution. The refined-score adapter imports this vendored package by default.
- `third_party/sembench` is a modified vendor snapshot of [SemBench](https://github.com/SemBench/SemBench). The original import did not record an upstream source revision or copy its top-level licenses. The upstream MIT and Apache texts are now retained with exact download provenance in `license_provenance.json`. That license revision does not identify the older modified source snapshot. Nested dependencies keep their own terms.
- SemBench benchmark data and human annotations have their original providers' terms. Source code licensing does not transfer ownership of input images, reviews or tables. Published reference GT already present in this repository is preserved; additional private data is excluded.
- [LRO-Bench](https://github.com/LROBench/LROBench) data is obtained separately. Its databases and annotations are not re-licensed by PROBE.
- Model weights are downloaded separately from their respective Hugging Face repositories. Their licenses and access requirements apply independently.

The missing original SemBench source revision remains a provenance gap. The original research checkout, historical outputs and Git history were retained; this release does not rewrite their history or assert that every historical result uses the current paper protocol.
