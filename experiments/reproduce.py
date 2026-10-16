"""Plan and execute isolated, serial paper runs; dry-run uses the standard library only."""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/paper_manifest.json"


def source_identity():
    files = [*ROOT.glob("*.py"), *ROOT.joinpath("experiments").glob("*.py")]
    analysis = {'report.py', 'paper_plots.py', 'paper_statistics.py'}
    files = [p for p in files if p.name not in analysis]
    files += [p for name in ("bargain", "lotus", "sembench") for p in ROOT.joinpath("third_party", name).rglob("*.py")]
    files += [MANIFEST, ROOT / "docs/artifact/manuscript_snapshot.json"]
    files += [ROOT/name for name in ("requirements.core.txt", "requirements.client.txt", "requirements.client.lock.txt")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files)}
    return hashes


def input_identity(args):
    """Hash prepared inputs before resume decisions, outside measured execution."""
    roots = dict(sembench=(args.data_root / 'SemBench' / 'files').resolve(),
                 lro=(args.lro_root or args.data_root / 'databases').resolve())
    entries = {}
    for label, root in roots.items():
        for path in sorted(root.rglob('*')):
            if not path.is_file() or any(p.startswith('.') for p in path.relative_to(root).parts):
                continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(chunk)
            entries[label+'/'+str(path.relative_to(root))] = digest.hexdigest()
    if not entries:
        raise ValueError('No prepared input files found')
    payload = json.dumps(entries, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'inputs-{digest}.json').write_bytes(payload+b'\n')
    return dict(sha256=digest, file_count=len(entries), roots={k:str(v) for k,v in roots.items()})


def _points_for_system(experiment, system):
    """Canonical per-system grid, also used by reporting for coverage."""
    points = experiment["points"]
    if system == "no_proxy":
        # Each expert must have its own matched full-oracle reference.
        if any("expert" in p for p in points):
            return [dict(p) for p in points]
        return [dict(points[0])]
    if system.startswith("BARGAIN") and any("calibrator" in p for p in points):
        return [dict(budget_pct=10, target=t) for t in (.1, .3, .5, .7, .9)]
    if system.startswith("BARGAIN") and len(points) > 1 and all("target" in p for p in points):
        return [dict(budget_pct=10, target=p["target"]) for p in points]
    if system in ("LOTUS", "ScaleDoc"):
        return [dict(p, budget_pct=10) for p in points]
    return [dict(p) for p in points]


def points_for_system(experiment, system):
    # Fixed-target baselines can map multiple PROBE budget points to one arm.
    # Do not execute or require coverage for duplicate canonical configurations.
    unique = {}
    for point in _points_for_system(experiment, system):
        unique.setdefault(json.dumps(point, sort_keys=True), point)
    return list(unique.values())


def model_settings(args):
    config = getattr(args, "model_config", None)
    return json.loads(config.read_text()) if config else {}


def plan(args):
    manifest = json.loads(MANIFEST.read_text())
    selected = list(manifest["experiments"]) if args.experiment == "all" else [args.experiment]
    if args.seeds is not None and (len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds)):
        raise ValueError("Seeds must be unique nonnegative integers")
    valid_queries = {q for name in selected for q in manifest["experiments"][name]["queries"]}
    valid_systems = {s for name in selected for s in manifest["experiments"][name]["systems"]}
    if set(args.queries or ()) - valid_queries:
        raise ValueError(f"Queries outside selected experiments: {sorted(set(args.queries) - valid_queries)}")
    if set(args.systems or ()) - valid_systems:
        raise ValueError(f"Systems outside selected experiments: {sorted(set(args.systems) - valid_systems)}")
    models = model_settings(args)
    jobs = []
    for name in selected:
        experiment = manifest["experiments"][name]
        queries = [q for q in experiment["queries"] if not args.queries or q in args.queries]
        systems = [s for s in experiment["systems"] if not args.systems or s in args.systems]
        seeds = args.seeds if args.seeds is not None else experiment["seeds"]
        for query, system, seed in itertools.product(queries, systems, seeds):
            for point in points_for_system(experiment, system):
                if getattr(args, "alphas", None) and point.get("alpha") not in args.alphas:
                    continue
                if getattr(args, "budgets", None) and point.get("budget_pct", 10) not in args.budgets:
                    continue
                if getattr(args, "representations", None) and point.get("representation") not in args.representations:
                    continue
                if getattr(args, "experts", None) and point.get("expert") not in args.experts:
                    continue
                expert = models.get("experts", {}).get(point.get("expert"), {})
                representation = models.get("representations", {}).get(point.get("representation"), {})
                workload = manifest["workloads"][query]
                jobs.append(dict(experiment=name, query=query, system=system, seed=seed,
                    operator=workload['operator'], modality=workload['modality'],
                    paper_N=workload["N"], data_environment=workload["environment"],
                    model=expert.get("model", args.model),
                    oracle_url=expert.get("oracle_url", args.oracle_url),
                    model_revision=expert.get("revision", args.model_revision) if "expert" not in point else expert.get("revision"),
                    ce_revision=args.ce_revision, representation=representation,
                    expert_configured="expert" not in point or all(expert.get(k) for k in ("model", "oracle_url", "revision")),
                    proxy=models.get("lotus_proxy", {}),
                    baseline_representation=models.get("baseline_representation", {}),
                    ablation_single_class=getattr(args, "ablation_single_class", "constant"),
                    protocol=dict(manifest["protocol"], ablation_single_class=(
                        getattr(args, "ablation_single_class", "constant") if name in ('sampling','refinement','trace') else 'constant')),
                    point=point, metric=workload["metric"],
                    executor=experiment["executors"].get(system),
                    timing="measured Spark end-to-end; downloads excluded; private client caches"))
    if not jobs:
        raise ValueError("The selected filters produce no experiment jobs")
    return jobs


def environment(job, args, output):
    # Resolve all paper knobs explicitly; unrelated historical LASO overrides cannot leak in.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LASO_", "ACTIVESAMPLING_", "PROBE_", "LLMSQL_", "SEMBENCH_"))}
    env.pop('ECOMM_Q7_TOPK', None)
    env.update(job["data_environment"])
    env["LROBENCH_DATABASES_DIR"] = str((args.lro_root or args.data_root / "databases").resolve())
    point = job["point"]
    env["PROBE_PAPER_JOB"] = json.dumps(job, sort_keys=True)
    env["PROBE_PAPER_OUTPUT"] = str(output)
    env["PROBE_DEVICE"] = args.device
    if getattr(args, "scaledoc_source", None):
        env["PROBE_SCALEDOC_SOURCE"] = str(args.scaledoc_source.resolve())
    env["PROBE_FROZEN_ROOT"] = str((args.output / "frozen").resolve())
    env["LLMSQL_EMBED_CACHE_DIR"] = str(output / "embed_cache")
    env["LLMSQL_KNN_CACHE_DIR"] = str(output / "graph_cache")
    sampler = point.get("sampler", "activeSampling")
    label = sampler + ("+refine+learned_calibration" if point.get("refine", True) else "+lr_only")
    env.update(LASO_DATA_ROOT=str(args.data_root.resolve()), LLMSQL_DATA_ROOT=str(args.data_root.resolve()),
               PROBE_DATA_ROOT=str(args.data_root.resolve()), LASO_QUERIES=job["query"],
               LASO_ABLATIONS=label, LASO_RANDOM_SEED=str(job["seed"]),
               LASO_BUDGET_PCTS=str(point.get("budget_pct", 10)),
               LASO_QUERY_BUDGET_JSON=json.dumps({job["query"]: [point.get("budget_pct", 10)]}),
               LASO_RESULTS_DIR=str(output), LASO_INCLUDE_NO_OPT="1" if job["system"] == "no_proxy" else "0",
               LASO_NO_OPT_BACKEND="python", SEMBENCH_MODEL=job["model"],
               SEMBENCH_ENABLE_THINKING="0", SEMBENCH_ORACLE_MAX_TOKENS="16",
               PROBE_ORACLE_URL=job["oracle_url"], PROBE_REPR="ce", PROBE_SAMPLING_BACKEND="fixed_ce",
               PROBE_SAMPLING_CONFIG=json.dumps({"protocol": "paper", "seed": job["seed"]}),
               PROBE_CE_CONFIG=json.dumps({"revision": job["ce_revision"], "device": args.device}),
               PROBE_CE_CACHE=str(output / "ce_cache"), LLMSQL_KNN_CACHE="0",
               PROBE_ORACLE_LOG=str(output / "oracle.jsonl"), PROBE_LABEL_CACHE="",
               PROBE_BLOCKING_K="0", PROBE_KNN_PROTOCOL="paper", PROBE_BARGAIN_MODE="",
               PROBE_CALIBRATION_PROTOCOL="paper", PROBE_ONLY_NO_OPT="0", SEMBENCH_UNPARSED_AS_FALSE="0",
               LASO_LC_SAMPLE_FRAC=".6", LASO_LC_CALIB_FRAC=".4", LASO_LC_ROUTE_FRAC="0",
               LASO_LC_BUDGET_SCOPE="preparation", LASO_LC_ROUTE_UNLIMITED="1",
               LASO_LC_CALIB_INPUT="refined", LASO_LC_ALPHA=str(point.get("alpha", .5)),
               LASO_LC_CALIBRATOR=point.get("calibrator", "platt"),
               LASO_REFINE_ALPHA=".8", LASO_REFINE_BETA=".8", LASO_REFINE_T1="50", LASO_REFINE_T2="50",
               LASO_REFINE_THETA=".5")
    if job["system"].startswith("BARGAIN"):
        env.update(PROBE_BARGAIN_MODE=job["system"].split("-")[-1],
                   PROBE_BARGAIN_TARGET=str(point.get("target", .9)), PROBE_BARGAIN_DELTA=".1",
                   PROBE_BARGAIN_BUDGET_FRAC="1", PROBE_BARGAIN_R_BUDGET_FRAC="1")
    # All paper arms enter through the same stable-ID transport. The pipeline
    # executor implements ablations/baselines without changing benchmark scorers.
    if job["system"] != "no_proxy":
        env["LASO_ABLATIONS"] = "activeSampling+refine+learned_calibration"
    if point.get("trace_batch_size"):
        env["PROBE_SAMPLING_CONFIG"] = json.dumps(dict(protocol="paper", seed=job["seed"], batch_size=point["trace_batch_size"]))
    if point.get("representation") in ("jina-reranker-m0", "SigLIP", "CLIP-base"):
        representation = job["representation"]
        if point["representation"] == "jina-reranker-m0":
            env["PROBE_CE_CONFIG"] = json.dumps(dict(model="jinaai/jina-reranker-m0", revision=representation["revision"], device=args.device))
        else:
            env["PROBE_REPR"] = "biencoder"
            env["LASO_EMBEDDING_MODEL"] = representation["snapshot"]
    if job["system"] == "no_proxy":
        # run.py's no-opt-only selector avoids running an additional proxy arm.
        env["LASO_ABLATIONS"] = ""
        env["PROBE_ONLY_NO_OPT"] = "1"
    return env


def run_job(job, args):
    import importlib.metadata
    hashes = source_identity()
    client_environment = dict(python=sys.version, packages={d.metadata['Name']:d.version
        for d in importlib.metadata.distributions() if d.metadata['Name']})
    external_sources = {}
    if job["system"] == "ScaleDoc":
        source = args.scaledoc_source
        external_sources = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(source.joinpath("src").rglob("*.py"))}
    stable_job = {k:v for k,v in job.items() if k!='endpoint_observations'}
    identity = dict(job=stable_job, sources=hashes, external_sources=external_sources,
                    inputs=args.input_identity,
                    client_environment=client_environment, device=args.device, data_root=str(args.data_root.resolve()),
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
            process = subprocess.Popen([sys.executable, str(ROOT / "run.py"), job["query"]],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                process.wait(timeout=args.timeout_seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # Kill only this owned client group, including its Spark JVM.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError(f"Client interrupted or timed out; preserved log: {attempt / 'stdout.log'}")
        rows_file = attempt / (job["query"] + "_results.csv")
        if process.returncode or not rows_file.exists():
            raise RuntimeError(f"Run failed; preserved log: {attempt / 'stdout.log'}")
        rows = list(csv.DictReader(rows_file.open()))
        if job["system"] == "no_proxy":
            rows = [r for r in rows if r.get("mode") == "no_opt"]
        if len(rows) != 1:
            raise RuntimeError(f"Expected one run row, found {len(rows)}: {rows_file}")
        row = rows[0]
        try:
            validate_measurement(row)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"{error}; preserved log: {attempt / 'stdout.log'}") from error
        n = int(float(row["policy_total_rows"]))
        if n != job["paper_N"]:
            raise RuntimeError(f"Candidate-universe mismatch: observed {n}, paper {job['paper_N']}")
        if not list(attempt.glob("*_result.csv/part-*.csv")):
            raise RuntimeError("Missing raw predictions")
        outputs = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in attempt.rglob("*") if p.is_file()}
        record = dict(**identity, endpoint_observations=job.get('endpoint_observations',{}),
                      outputs=outputs, started_unix=started, ended_unix=time.time(),
                      row=row, run=key, evidence_status="measured_single_run",
                      expert_revision_status="declared; must match separately archived server metadata")
        tmp = completed.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n"); tmp.replace(completed)
        print(json.dumps(dict(status="complete", run=key)), flush=True)


def validate_measurement(row):
    if int(float(row.get("oracle_failures") or 0)):
        raise RuntimeError("Unresolved expert answers; this attempt is not complete")
    if not row.get("gt_source_sha256"):
        raise RuntimeError("Missing human-ground-truth provenance")
    for key in ("f1", "delegation_rate", "duration_s"):
        value = float(row[key])
        if not math.isfinite(value) or (value <= 0 if key == "duration_s" else not 0 <= value <= 1):
            raise RuntimeError(f"Invalid measured {key}: {row[key]}")


def immutable_revision(value):
    return bool(value and re.fullmatch(r"[0-9a-fA-F]{40}", value))


def endpoint_metadata(jobs):
    """Verify advertised model IDs; this cannot attest the served weight commit."""
    from urllib.request import urlopen
    observations = {}
    for job in jobs:
        bindings = [(job['oracle_url'], job['model'])]
        if job['system'] == 'LOTUS' and job['operator'] == 'filter':
            bindings.append((job['proxy']['oracle_url'], job['proxy']['model']))
        for url, model in bindings:
            if url not in observations:
                with urlopen(url.removesuffix('/chat/completions')+'/models', timeout=30) as response:
                    observations[url] = json.load(response)
            ids = {entry['id'] for entry in observations[url]['data']}
            if model not in ids:
                raise ValueError(f"Endpoint {url} does not advertise configured model {model}; available IDs: {sorted(ids)}")
    return observations


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
    parser.add_argument("--scaledoc-source", type=Path, help="Separately cloned upstream ScaleDoc checkout")
    parser.add_argument("--model-config", type=Path, help="JSON with experts, representations and lotus_proxy endpoints/revisions")
    parser.add_argument("--alphas", type=float, nargs="+", help="Select existing routing tradeoff points")
    parser.add_argument("--budgets", type=float, nargs="+", help="Select existing budget points; does not rewrite the paper grid")
    parser.add_argument("--representations", nargs="+", help="Select named representation arms")
    parser.add_argument("--experts", nargs="+", help="Select named expert arms")
    parser.add_argument("--ablation-single-class", choices=["constant", "oracle"], default="constant",
                        help="Ablation with one training class: constant prediction (default, no extra calls) or explicit S3 oracle variant")
    parser.add_argument("--oracle-url", default=os.environ.get("PROBE_ORACLE_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PROBE_DATA_ROOT", "data")))
    parser.add_argument("--lro-root", type=Path,
                        default=Path(os.environ["LROBENCH_DATABASES_DIR"]) if os.environ.get("LROBENCH_DATABASES_DIR") else None)
    parser.add_argument("--timeout-seconds", type=float, default=86400, help="Hard timeout for each owned Spark client job")
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip intact completed jobs; incomplete jobs restart as a preserved new attempt")
    args = parser.parse_args()
    try:
        jobs = plan(args)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(dict(profile="paper", count=len(jobs), jobs=jobs), indent=2)); return
    unavailable = sorted({j["system"] for j in jobs if j["executor"] is None})
    if unavailable:
        parser.error(f"Unaudited adapters: {unavailable}; see docs/artifact/experiment_alignment.md")
    prerequisites = []
    for job in jobs:
        if not immutable_revision(job["model_revision"]) or not job["expert_configured"]:
            prerequisites.append(f"{job['point'].get('expert', job['model'])}: expert endpoint and immutable revision")
        if job["point"].get("representation") in ("jina-reranker-m0", "SigLIP", "CLIP-base") and not immutable_revision(job["representation"].get("revision")):
            prerequisites.append(f"{job['point']['representation']}: immutable representation revision")
        if job["system"] == "LOTUS" and job['operator'] == 'filter' and (not all(job["proxy"].get(k) for k in ("model", "oracle_url")) or not immutable_revision(job["proxy"].get("revision"))):
            prerequisites.append("LOTUS: Qwen3-VL-2B proxy endpoint and immutable revision")
        if not immutable_revision(job["ce_revision"]):
            prerequisites.append("Cross-encoder: immutable 40-character commit revision")
    if any(j["system"] == "ScaleDoc" for j in jobs):
        source = args.scaledoc_source
        if not args.device.startswith('cuda'):
            prerequisites.append("ScaleDoc: CUDA client required by upstream augmentation")
        if source is None or not all((source / p).is_file() for p in (
                "src/model.py", "src/cascade.py", "src/utils/sample.py", "src/train_cl.py",
                "src/utils/train.py", "src/utils/log.py")):
            prerequisites.append("ScaleDoc: --scaledoc-source with the complete upstream src/ tree")
    if prerequisites:
        parser.error("Missing model prerequisites: " + "; ".join(sorted(set(prerequisites))))
    if not args.data_root.is_dir():
        parser.error("--data-root must contain the prepared, licensed benchmark data")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    try:
        observations = endpoint_metadata(jobs)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Model endpoint preflight failed: {error}")
    for job in jobs:
        job['endpoint_observations'] = {url:observations[url] for url in (
            [job['oracle_url']] + ([job['proxy']['oracle_url']] if job['system']=='LOTUS' and job['operator']=='filter' else []))}
    # Resolve model assets once before timing, never hide download time in a run.
    needed = set()
    for job in jobs:
        if job["system"] in ("PROBE", "BARGAIN-A", "BARGAIN-P", "BARGAIN-R") and job["point"].get("representation") not in ("SigLIP", "CLIP-base"):
            needed.add(("jinaai/jina-reranker-m0", job["representation"]["revision"]) if job["point"].get("representation") == "jina-reranker-m0" else ("Qwen/Qwen3-VL-Reranker-2B", job["ce_revision"]))
        if job["system"] in ("CSV", "ScaleDoc", "AIQA") or (job['system']=='LOTUS' and job['operator']=='join'):
            sys.path.insert(0, str(ROOT))
            from experiments.paper_baselines import BASELINE_MODELS
            kind = 'multimodal' if 'image' in job['modality'] else 'text'
            rep = dict(BASELINE_MODELS[kind], **job["baseline_representation"].get(kind, {}))
            needed.add((rep["model"], rep["revision"]))
    if needed:
        from huggingface_hub import snapshot_download
        for model, revision in sorted(needed):
            if not immutable_revision(revision):
                parser.error(f"{model}: representation revision must be an immutable 40-character commit")
            snapshot_download(model, revision=revision)
    for job in jobs:
        if job["point"].get("representation") in ("SigLIP", "CLIP-base"):
            from huggingface_hub import snapshot_download
            rep = job["representation"]
            rep["snapshot"] = snapshot_download(rep["model"], revision=rep["revision"])
    print('Hashing prepared dataset files before resume checks', flush=True)
    args.input_identity = input_identity(args)
    for index, job in enumerate(jobs, 1):
        print(json.dumps(dict(status="starting", index=index, total=len(jobs),
                              experiment=job["experiment"], query=job["query"],
                              system=job["system"], seed=job["seed"], point=job["point"])), flush=True)
        run_job(job, args)


if __name__ == "__main__":
    main()
