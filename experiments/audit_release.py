"""Scan publish candidates and reachable Git blobs; report locations, never secret values."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,})"),
    "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "google_api_key": re.compile(rb"AIza[A-Za-z0-9_-]{35}"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "personal_absolute_path": re.compile(rb"/(?:localhome|home|local-scratch/localhome)/[A-Za-z0-9_.-]+/"),
}


def findings(data):
    return [{"type": kind, "count": len(matches),
             "lines": sorted({data[:m.start()].count(b"\n") + 1 for m in matches})}
            for kind, pattern in PATTERNS.items() if (matches := list(pattern.finditer(data)))]


def candidates():
    files = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    for folder in (ROOT, ROOT / "tests", ROOT / "experiments"):
        files.update(str(p.relative_to(ROOT)) for p in folder.glob("*.py"))
    for folder in (ROOT / "docs/artifact",):
        files.update(str(p.relative_to(ROOT)) for p in folder.rglob("*") if p.is_file())
    files.update(["requirements.core.txt", "requirements.client.txt", "requirements.client.lock.txt", "NOTICE.md",
                  "experiments/paper_manifest.json", "third_party/sembench/LICENSE-MIT",
                  "third_party/sembench/LICENSE-APACHE", "third_party/sembench/license_provenance.json"])
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/release_audit.json"))
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    current = []; inventory = {}
    for name in candidates():
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        inventory[name] = hashlib.sha256(data).hexdigest()
        if hit := findings(data):
            current.append(dict(file=name, findings=hit))
    historical = []; n_blobs = 0
    if args.history:
        entries = subprocess.check_output(["git", "rev-list", "--objects", "--all"], cwd=ROOT).splitlines()
        proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        for entry in entries:
            oid, _, name = entry.partition(b" ")
            proc.stdin.write(oid + b"\n"); proc.stdin.flush()
            header = proc.stdout.readline().split()
            size = int(header[2]); data = proc.stdout.read(size); proc.stdout.read(1)
            if header[1] != b"blob":
                continue
            n_blobs += 1
            if hit := findings(data):
                historical.append(dict(blob=oid.decode(), file=name.decode(errors="replace"), findings=hit))
        proc.stdin.close(); proc.wait()
    result = dict(scope="Selected publish sources/records and all reachable refs when --history; pattern scan is not an exhaustive secret detector",
                  current_findings=current, historical_findings=historical, reachable_blobs_scanned=n_blobs,
                  publish_files=inventory, policy="No secret values emitted; no automatic history rewrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(publish_files=len(inventory), current_files_with_findings=len(current),
                          historical_blobs_with_findings=len(historical), reachable_blobs_scanned=n_blobs)))


if __name__ == "__main__":
    main()
