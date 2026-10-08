"""Validate immutable run records, then aggregate and plot without workbook overrides."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load_runs(root, manifest):
    records = []
    for path in sorted(root.glob("*/completed.json")):
        run = json.loads(path.read_text()); job = run["job"]
        if job["protocol"] != manifest["protocol"]:
            raise ValueError(f"Protocol mismatch: {path}")
        for file, expected in run["outputs"].items():
            p = path.parent / file
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Missing or modified raw output: {p}")
        row = run["row"]
        n = int(float(row["policy_total_rows"]))
        if n != job["paper_N"]:
            raise ValueError(f"Universe mismatch: {path}")
        quality, delegation, latency = (float(row[k]) for k in ["f1", "delegation_rate", "duration_s"])
        if not (0 <= quality <= 1 and 0 <= delegation <= 1 and latency > 0):
            raise ValueError(f"Invalid measurement: {path}")
        if int(float(row.get("oracle_failures") or 0)):
            raise ValueError(f"Unresolved oracle failures: {path}")
        if not row.get("gt_source_sha256"):
            raise ValueError(f"Missing GT provenance: {path}")
        records.append(dict(query=job["query"], system=job["system"], experiment=job["experiment"],
                            seed=job["seed"], point=json.dumps(job["point"], sort_keys=True),
                            model=job["model"], metric=job["metric"], N=n,
                            f1=quality, delegation=delegation, latency=latency,
                            raw_record=str(path.relative_to(root)),
                            expert_revision_status=run["expert_revision_status"]))
    return records


def aggregate(records, expected_seeds):
    grouped = {}
    keys = ("experiment", "query", "system", "model", "metric", "point")
    for row in records:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        seeds = [r["seed"] for r in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds within a configuration: {key}")
        expected = expected_seeds[key[0]]
        complete = set(seeds) == set(expected)
        values = [r["f1"] for r in rows]
        mean = statistics.mean(values)
        # CI is only emitted for the prescribed five-run main quality groups.
        ci = None
        if key[0] == "main" and complete and len(values) == 5:
            ci = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)
        output.append(dict(zip(keys, key), seeds=sorted(seeds), expected_seeds=expected,
                           complete=complete, n_runs=len(rows), f1_mean=mean, f1_ci95_halfwidth=ci,
                           delegation_mean=statistics.mean(r["delegation"] for r in rows),
                           latency_mean_s=statistics.mean(r["latency"] for r in rows)))
    return output


def geometric_mean(values):
    if not values or any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError("Geometric mean requires a complete finite nonnegative population")
    return 0. if any(v == 0 for v in values) else math.exp(statistics.mean(math.log(v) for v in values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("results/paper"))
    p.add_argument("--output", type=Path, default=Path("results/figures"))
    p.add_argument("--plot-all", action="store_true")
    a = p.parse_args()
    manifest = json.loads((ROOT / "experiments/paper_manifest.json").read_text())
    records = load_runs(a.input, manifest)
    aggregated = aggregate(records, {k: v["seeds"] for k, v in manifest["experiments"].items()})
    a.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("validated_runs", records), ("aggregate", aggregated)]:
        (a.output / (name + ".json")).write_text(json.dumps(rows, indent=2) + "\n")
        if rows:
            with (a.output / (name + ".csv")).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    coverage = {}
    for name, experiment in manifest["experiments"].items():
        present = [r for r in aggregated if r["experiment"] == name and r["complete"]]
        pairs = {(r["query"], r["system"], r["point"]) for r in present}
        expected = set()
        for query, system in __import__("itertools").product(experiment["queries"], experiment["systems"]):
            points = experiment["points"][:1] if system == "no_proxy" else experiment["points"]
            for point in points:
                if name == "calibration" and system.startswith("BARGAIN"):
                    if point.get("calibrator") != "platt" or point["target"] not in (.1, .3, .5, .7, .9):
                        continue
                expected.add((query, system, json.dumps(point, sort_keys=True)))
        coverage[name] = dict(complete_configuration_groups=len(pairs & expected),
                              expected_configuration_groups=len(expected),
                              missing_pairs=sorted(expected - pairs),
                              status="partial" if pairs != expected else "query_system_coverage_only")
    (a.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    claims = {}
    main = manifest["experiments"]["main"]
    for system in main["systems"]:
        group = [r for r in aggregated if r["experiment"] == "main" and
                 r["system"] == system and r["complete"]]
        if len(group) == len(main["queries"]) and {r["query"] for r in group} == set(main["queries"]):
            claims[system] = dict(query_coverage=main["queries"],
                                  f1_geomean=geometric_mean([r["f1_mean"] for r in group]),
                                  delegation_geomean=geometric_mean([r["delegation_mean"] for r in group]),
                                  delegation_median=statistics.median(r["delegation_mean"] for r in group),
                                  latency_geomean_s=geometric_mean([r["latency_mean_s"] for r in group]),
                                  status="complete seed/query coverage; expert revision and controlled timing need separate attestation")
    (a.output / "recomputed_claims.json").write_text(json.dumps(claims, indent=2) + "\n")
    if a.plot_all:
        # Refuse incomplete grids instead of silently omitting queries or inventing error bars.
        missing = [k for k, v in coverage.items() if v["status"] == "partial"]
        if missing:
            p.error(f"Cannot plot the full paper: incomplete evidence for {missing}; see coverage.json")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name in manifest["experiments"]:
            rows = [r for r in aggregated if r["experiment"] == name]
            fig, ax = plt.subplots(figsize=(10, 4))
            for system in sorted({r["system"] for r in rows}):
                arm = [r for r in rows if r["system"] == system]
                ax.scatter([r["delegation_mean"] for r in arm], [r["f1_mean"] for r in arm], label=system)
            ax.set(xlabel="Unique candidate delegation fraction", ylabel="Query scorer F1 against GT")
            ax.legend(); fig.tight_layout()
            fig.savefig(a.output / (name + "_quality_delegation_diagnostic.pdf")); plt.close(fig)
    print(json.dumps(dict(validated_runs=len(records), groups=len(aggregated), coverage=coverage)))


if __name__ == "__main__":
    main()
