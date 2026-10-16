"""Small real-model public Spark filter/join validation; no paper-quality claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/smoke"))
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--oracle-url", default=os.environ.get("PROBE_ORACLE_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(dict(profile="smoke", N_per_operation=4, preparation_budget=a.budget,
                             sampling_budget=3*a.budget//5, operations=["filter", "join"],
                             expert=a.model, oracle_url=a.oracle_url, device=a.device,
                             ce_revision="4bd860ac4f15ad1897a214615cccc700f8f71818"), indent=2)); return
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ["PROBE_CE_CACHE"] = str((a.output / "ce_cache").resolve())
    os.environ["PROBE_ORACLE_LOG"] = str((a.output / "oracle.jsonl").resolve())
    os.environ["PROBE_REPR"] = "ce"
    os.environ["LLMSQL_KNN_CACHE"] = "0"
    sys.path.insert(0, str(ROOT))
    from pyspark.sql import SparkSession
    import sem_ops_register
    import proxy_model as pm
    from ce_ranker import CEConfig
    from learned_calibration import paper_calibration_config
    cfg = paper_calibration_config(a.budget)
    options = dict(model=a.model, url=a.oracle_url, embedding_model="unused-in-ce-mode",
                   proxy_sample=True, sample_size=cfg.sample_budget, budget_size=a.budget,
                   learned_calibration_config=cfg,
                   ce_config=CEConfig(revision="4bd860ac4f15ad1897a214615cccc700f8f71818",
                                      batch_size=2, device=a.device))
    spark = (SparkSession.builder.master("local[1]").appName("probe-paper-smoke")
             .config("spark.ui.enabled", "false").config("spark.driver.host", "127.0.0.1")
             .config("spark.sql.shuffle.partitions", "1")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    records = []
    try:
        for kind in ("filter", "join"):
            started = time.perf_counter()
            if kind == "filter":
                df = spark.range(4).selectExpr("id", "CASE WHEN id%2=0 THEN 'a red car' ELSE 'a blue bicycle' END AS text")
                answer = df.sem_filter(prompt="Does this describe a car? {text}", **options)
            else:
                left = spark.range(2).selectExpr("id AS left_id", "CASE WHEN id=0 THEN 'red car' ELSE 'blue bicycle' END AS left_text")
                right = spark.range(2).selectExpr("id AS right_id", "CASE WHEN id=0 THEN 'automobile' ELSE 'bicycle' END AS right_text")
                answer = left.sem_join(right=right, prompt="Same kind of vehicle? {left_text} {right_text}", **options)
            rows = [r.asDict() for r in answer.collect()]
            profile = pm.get_last_profiling()
            oracle = profile["oracle_budget"]
            assert oracle["candidate_attempts"] > 0 and oracle["failures"] == 0
            assert oracle["preparation_attempts"] <= a.budget
            assert oracle["unique_labels"] == 4  # one-class tiny L triggers S3, retaining L
            assert profile["cluster_profiling"]["whole_query_fallback"]
            record = dict(kind=kind, N=4, output=rows, oracle=oracle,
                          ce=profile["ce_provenance"], seconds=time.perf_counter()-started)
            records.append(record)
            print(json.dumps(record, default=str), flush=True)
        sources = [p for p in ROOT.glob("*.py")] + [Path(__file__)]
        result = dict(profile="smoke", runs=records,
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      scope="Real CE and Gemma expert, text fixtures, public Spark filter/join and S3. No population-quality or timing comparison claim.")
        (a.output / "validation.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
