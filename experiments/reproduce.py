"""Plan and execute isolated, serial paper runs; dry-run uses the standard library only."""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/paper_manifest.json"


def source_identity():
    files = [*ROOT.glob("*.py"), *ROOT.joinpath("experiments").glob("*.py")]
    files += [MANIFEST, ROOT / "docs/artifact/manuscript_snapshot.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files)}
    return hashes


def plan(args):
    manifest = json.loads(MANIFEST.read_text())
    if args.experiment == "all":
        jobs = []
        for name in manifest["experiments"]:
            part = argparse.Namespace(**vars(args))
            part.experiment = name
            jobs.extend(plan(part))
        return jobs
    experiment = manifest["experiments"][args.experiment]
    queries = args.queries or experiment["queries"]
    systems = args.systems or experiment["systems"]
    unknown = set(queries) - manifest["workloads"].keys()
    if unknown:
        raise ValueError(f"Unknown queries: {sorted(unknown)}")
    unknown = set(systems) - set(experiment["systems"])
    if unknown:
        raise ValueError(f"Systems outside this experiment: {sorted(unknown)}")
    seeds = args.seeds if args.seeds is not None else experiment["seeds"]
    jobs = []
    for query, system, seed in itertools.product(queries, systems, seeds):
        for point in experiment["points"]:
            if args.experiment == "calibration" and system.startswith("BARGAIN"):
                if point.get("calibrator") != "platt" or point["target"] not in (.1, .3, .5, .7, .9):
                    continue
            if system == "no_proxy" and point != experiment["points"][0]:
                continue
            workload = manifest["workloads"][query]
            jobs.append(dict(experiment=args.experiment, query=query, system=system, seed=seed,
                             paper_N=workload["N"], data_environment=workload["environment"],
                             model=args.model, oracle_url=args.oracle_url,
                             model_revision=args.model_revision, ce_revision=args.ce_revision,
                             protocol=manifest["protocol"], point=point,
                             metric=workload["metric"],
                             executor=experiment["executors"].get(system),
                             timing="measured Spark end-to-end; model download excluded; isolated client caches"))
    return jobs


def environment(job, args, output):
    # Resolve all paper knobs explicitly; unrelated historical LASO overrides cannot leak in.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LASO_", "ACTIVESAMPLING_"))}
    env.update(job["data_environment"])
    env["LROBENCH_DATABASES_DIR"] = str((args.lro_root or args.data_root / "databases").resolve())
    point = job["point"]
    sampler = point.get("sampler", "activeSampling")
    label = sampler + ("+refine+learned_calibration" if point.get("refine", True) else "+lr_only")
    env.update(LASO_DATA_ROOT=str(args.data_root.resolve()), LLMSQL_DATA_ROOT=str(args.data_root.resolve()),
               PROBE_DATA_ROOT=str(args.data_root.resolve()), LASO_QUERIES=job["query"],
               LASO_ABLATIONS=label, LASO_RANDOM_SEED=str(job["seed"]),
               LASO_BUDGET_PCTS=str(point.get("budget_pct", 10)),
               LASO_QUERY_BUDGET_JSON=json.dumps({job["query"]: [point.get("budget_pct", 10)]}),
               LASO_RESULTS_DIR=str(output), LASO_INCLUDE_NO_OPT="1" if job["system"] == "no_proxy" else "0",
               LASO_NO_OPT_BACKEND="python", SEMBENCH_MODEL=job["model"],
               PROBE_ORACLE_URL=job["oracle_url"], PROBE_REPR="ce", PROBE_SAMPLING_BACKEND="fixed_ce",
               PROBE_SAMPLING_CONFIG=json.dumps({"protocol": "paper", "seed": job["seed"]}),
               PROBE_CE_CONFIG=json.dumps({"revision": job["ce_revision"], "device": args.device}),
               PROBE_CE_CACHE=str(output / "ce_cache"), LLMSQL_KNN_CACHE="0",
               PROBE_ORACLE_LOG=str(output / "oracle.jsonl"), PROBE_LABEL_CACHE="",
               PROBE_BLOCKING_K="0", PROBE_KNN_PROTOCOL="paper", PROBE_BARGAIN_MODE="",
               PROBE_CALIBRATION_PROTOCOL="paper", PROBE_ONLY_NO_OPT="0", SEMBENCH_UNPARSED_AS_FALSE="0",
               LASO_LC_SAMPLE_FRAC=".8", LASO_LC_CALIB_FRAC=".2", LASO_LC_ROUTE_FRAC="0",
               LASO_LC_BUDGET_SCOPE="preparation", LASO_LC_ROUTE_UNLIMITED="1",
               LASO_LC_CALIB_INPUT="refined", LASO_LC_ALPHA=str(point.get("alpha", .5)),
               LASO_LC_CALIBRATOR=point.get("calibrator", "platt"),
               LASO_REFINE_ALPHA=".8", LASO_REFINE_BETA=".8", LASO_REFINE_T1="50", LASO_REFINE_T2="50",
               LASO_REFINE_THETA=".5")
    if job["system"].startswith("BARGAIN"):
        env.update(PROBE_BARGAIN_MODE=job["system"].split("-")[-1],
                   PROBE_BARGAIN_TARGET=str(point.get("target", .9)), PROBE_BARGAIN_DELTA=".1")
    if job["system"] == "AIQA":
        env["LASO_ABLATIONS"] = "uqe_active+lr_only"
        env["PROBE_REPR"] = "mixed"
    if job["system"] == "no_proxy":
        # run.py's no-opt-only selector avoids running an additional proxy arm.
        env["LASO_ABLATIONS"] = ""
        env["PROBE_ONLY_NO_OPT"] = "1"
    return env


def run_job(job, args):
    hashes = source_identity()
    identity = dict(job=job, sources=hashes, data_root=str(args.data_root.resolve()),
                    lro_root=str((args.lro_root or args.data_root / "databases").resolve()))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = args.output.resolve() / key
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = folder / "completed.json"
        if completed.exists():
            record = json.loads(completed.read_text())
            intact = all((folder / p).is_file() and hashlib.sha256((folder / p).read_bytes()).hexdigest() == h
                         for p, h in record["outputs"].items())
            if args.resume and intact:
                print(json.dumps(dict(status="already_complete", run=key)), flush=True)
                return
            raise RuntimeError(f"Existing completed run {key}; use --resume with intact outputs")
        attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))) + 1:03d}"
        attempt.mkdir()
        env = environment(job, args, attempt)
        (attempt / "resolved.json").write_text(json.dumps(identity, indent=2) + "\n")
        started = time.time()
        with (attempt / "stdout.log").open("w") as log:
            process = subprocess.run([sys.executable, str(ROOT / "run.py"), job["query"]],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rows_file = attempt / (job["query"] + "_results.csv")
        if process.returncode or not rows_file.exists():
            raise RuntimeError(f"Run failed; preserved log: {attempt / 'stdout.log'}")
        rows = list(csv.DictReader(rows_file.open()))
        if job["system"] == "no_proxy":
            rows = [r for r in rows if r.get("mode") == "no_opt"]
        if len(rows) != 1:
            raise RuntimeError(f"Expected one run row, found {len(rows)}: {rows_file}")
        row = rows[0]
        n = int(float(row["policy_total_rows"]))
        if n != job["paper_N"]:
            raise RuntimeError(f"Candidate-universe mismatch: observed {n}, paper {job['paper_N']}")
        if not list(attempt.glob("*_result.csv/part-*.csv")):
            raise RuntimeError("Missing raw predictions")
        outputs = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in attempt.rglob("*") if p.is_file()}
        record = dict(**identity, outputs=outputs, started_unix=started, ended_unix=time.time(),
                      row=row, run=key, evidence_status="measured_single_run",
                      expert_revision_status="declared; must match separately archived server metadata")
        tmp = completed.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n"); tmp.replace(completed)
        print(json.dumps(dict(status="complete", run=key)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = list(json.loads(MANIFEST.read_text())["experiments"])
    parser.add_argument("--experiment", default="main", choices=["all", *names])
    parser.add_argument("--queries", nargs="+")
    parser.add_argument("--systems", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--model-revision", default=os.environ.get("PROBE_ORACLE_REVISION"))
    parser.add_argument("--ce-revision", default="4bd860ac4f15ad1897a214615cccc700f8f71818")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle-url", default=os.environ.get("PROBE_ORACLE_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PROBE_DATA_ROOT", "data")))
    parser.add_argument("--lro-root", type=Path,
                        default=Path(os.environ["LROBENCH_DATABASES_DIR"]) if os.environ.get("LROBENCH_DATABASES_DIR") else None)
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip intact completed jobs; incomplete jobs restart as a preserved new attempt")
    args = parser.parse_args()
    jobs = plan(args)
    if args.dry_run:
        print(json.dumps(dict(profile="paper", count=len(jobs), jobs=jobs), indent=2)); return
    unavailable = sorted({j["system"] for j in jobs if j["executor"] is None})
    if unavailable:
        parser.error(f"Unaudited adapters: {unavailable}; see docs/artifact/experiment_alignment.md")
    if not args.model_revision:
        parser.error("--model-revision is required for a measured paper run")
    if not args.data_root.is_dir():
        parser.error("--data-root must contain the prepared, licensed benchmark data")
    for job in jobs:
        run_job(job, args)


if __name__ == "__main__":
    main()
