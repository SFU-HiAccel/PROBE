"""
ROxford (Oxford Buildings) Ground Truth Generator for JoinML-style Pairwise Join Tasks
========================================================================================

Paper context (arXiv:2603.16153):
    Query: SELECT COUNT(*)
           FROM oxbuild_gallery AS g
           JOIN oxbuild_queries AS q
           ON NL('Two images show the same landmark: {g.image} {q.image}')

The ROxford dataset (Revisited Oxford) uses per-query relevance lists with three labels:
    - 'easy'  : clearly the same landmark (high confidence positive)
    - 'hard'  : same landmark but harder to match
    - 'junk'  : ambiguous / unclear — conventionally excluded from evaluation

Three evaluation protocols exist:
    - Easy   (E): positives = easy only
    - Medium (M): positives = easy + hard
    - Hard   (H): positives = hard only

This script converts the retrieval-format ground truth (per-query relevance lists) into
pairwise join format: (gallery_img, query_img, label) where label ∈ {0, 1}.

Download:
    Dataset images + gnd_roxford5k.pkl are auto-downloaded from the official source:
    https://cmp.felk.cvut.cz/revisitop/
    via the revisitop utility (or manually).

Usage:
    python roxford_groundtruth.py --data_dir ./data/roxford5k --protocol medium
    python roxford_groundtruth.py --data_dir ./data/roxford5k --protocol easy --output gt_easy.csv
"""
import os as _os  # noqa: E402  (anonymized path config)
DATA_ROOT = _os.environ.get("LLMSQL_DATA_ROOT", _os.path.expanduser("~"))


import os
import pickle
import argparse
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd


# ── Download URLs ──────────────────────────────────────────────────────────────
DATASET_URL    = "https://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl"
IMAGES_URL     = "https://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/jpg.tar.gz"
IMAGES_URL_ALT = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz"

# revisitop1m distractor pack (1M negative images) — needed to reproduce the
# JoinML paper's Table 2 stats (|L|=70, |R|=1e6, selectivity 7.8e-5). We only
# download the filename manifest (~50 MB plain text) by default; actual image
# tarballs are multi-GB and optional. Source: filipradenovic/revisitop
# (python/download.py -> download_distractors).
R1M_MANIFEST_URL = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/revisitop1m.txt"


def download_file(url: str, dest: Path, description: str = "") -> None:
    """Download a file with a progress indicator."""
    print(f"Downloading {description or url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


def download_roxford(data_dir: Path, skip_images: bool = False) -> None:
    """
    Download ROxford dataset files into data_dir.

    Structure after download:
        data_dir/
            gnd_roxford5k.pkl
            jpg/
                all_souls_000000.jpg
                ...
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground truth pickle
    gnd_path = data_dir / "gnd_roxford5k.pkl"
    if not gnd_path.exists():
        download_file(DATASET_URL, gnd_path, "gnd_roxford5k.pkl")
    else:
        print(f"Found existing: {gnd_path}")

    # 2. Images (optional — not needed for GT construction, but needed for model eval)
    if not skip_images:
        jpg_dir = data_dir / "jpg"
        if jpg_dir.exists() and any(jpg_dir.iterdir()):
            print(f"Images already present in {jpg_dir}, skipping download.")
        else:
            tgz_path = data_dir / "jpg.tar.gz"
            if not tgz_path.exists():
                download_file(IMAGES_URL, tgz_path, "roxford5k images (~1.8 GB)")
            print("Extracting images ...")
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(data_dir)
            print(f"Images extracted to {jpg_dir}")


def download_r1m_tarballs(data_dir: Path, n_tarballs: int = 100,
                          start: int = 1, extract: bool = True,
                          keep_archive: bool = False) -> None:
    """Download and optionally extract revisitop1m image tarballs.

    The revisitop1m distractor set is split across 100 tarballs named
    revisitop1m.{1..100}.tar.gz under
    http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/. Each tarball is
    ~4 GB and contains ~10,000 images. Full set is ~400 GB / 1M images.

    Parameters
    ----------
    data_dir    : root under which revisitop1m/jpg/ will be populated
    n_tarballs  : how many tarballs to fetch (1 to 100)
    start       : index of the first tarball (1-based, matches upstream numbering)
    extract     : if True, un-tar each archive after download
    keep_archive: if False, delete each .tar.gz after successful extraction
    """
    import tarfile
    r1m_dir = data_dir / "revisitop1m"
    jpg_dir = r1m_dir / "jpg"
    r1m_dir.mkdir(parents=True, exist_ok=True)

    for i in range(start, start + n_tarballs):
        if i < 1 or i > 100:
            print(f"Skipping out-of-range tarball index {i} (valid: 1..100)")
            continue
        url = f"http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.{i}.tar.gz"
        tgz_path = r1m_dir / f"revisitop1m.{i}.tar.gz"
        if tgz_path.exists():
            print(f"[{i:3d}/100] already present: {tgz_path.name}")
        else:
            print(f"[{i:3d}/100] downloading {url}")
            download_file(url, tgz_path, f"revisitop1m.{i}.tar.gz (~4 GB)")

        if extract:
            print(f"[{i:3d}/100] extracting into {jpg_dir} ...")
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(r1m_dir)
            if not keep_archive:
                tgz_path.unlink()
                print(f"[{i:3d}/100] removed archive {tgz_path.name}")


def load_r1m_distractors(data_dir: Path, n_distractors: int,
                         download: bool = True) -> list:
    """Load the revisitop1m distractor filename list.

    Returns a list of up to n_distractors filenames (with the .jpg suffix
    already present — that's how the upstream manifest stores them).
    These are added to the gallery as label=0 rows so that |R| hits the
    paper's 1M target and selectivity drops to 7.8e-5.

    If download=True, fetches the ~50 MB plain-text filename manifest
    (revisitop1m.txt) on first use. Actual image tarballs are NOT
    downloaded here (multi-GB each, 100 of them) — paths in the manifest
    are referenced as <data_dir>/revisitop1m/jpg/<name>.jpg and only
    resolved when the grid actually loads the images.
    """
    import gzip

    r1m_dir = data_dir / "revisitop1m"
    manifest = r1m_dir / "revisitop1m.txt"
    manifest_gz = r1m_dir / "revisitop1m.txt.gz"

    if not manifest.exists() and not manifest_gz.exists():
        if not download:
            raise FileNotFoundError(
                f"No revisitop1m manifest under {r1m_dir}. "
                "Run with --download or manually place revisitop1m.txt there."
            )
        download_file(R1M_MANIFEST_URL, manifest, "revisitop1m manifest")

    names = []
    if manifest.exists():
        with open(manifest, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.append(line)
                if len(names) >= n_distractors:
                    break
    else:
        with gzip.open(manifest_gz, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.append(line)
                if len(names) >= n_distractors:
                    break
    return names[:n_distractors]


def load_gnd(data_dir: Path) -> dict:
    """Load the ground truth pickle file."""
    gnd_path = data_dir / "gnd_roxford5k.pkl"
    if not gnd_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {gnd_path}\n"
            "Run with --download to fetch it automatically."
        )
    with open(gnd_path, "rb") as f:
        gnd_data = pickle.load(f)
    return gnd_data


def build_pairwise_gt(
    gnd_data: dict,
    protocol: str = "medium",
    include_junk: bool = False,
) -> pd.DataFrame:
    """
    Convert ROxford retrieval GT into pairwise join format.

    Parameters
    ----------
    gnd_data    : dict loaded from gnd_roxford5k.pkl
                  Keys: 'imlist' (gallery), 'qimlist' (queries), 'gnd' (per-query annots)
    protocol    : 'easy' | 'medium' | 'hard'
                  easy   -> positives = easy only
                  medium -> positives = easy + hard  (default, matches paper selectivity)
                  hard   -> positives = hard only
    include_junk: if True, junk pairs are included with label=-1 (ambiguous)
                  if False (default), junk pairs are excluded entirely

    Returns
    -------
    DataFrame with columns:
        gallery_id    : index into gnd_data['imlist']
        query_id      : index into gnd_data['qimlist']
        gallery_img   : filename stem of gallery image (e.g. 'all_souls_000000')
        query_img     : filename stem of query image
        label         : 1 = match (same landmark), 0 = non-match, -1 = junk (if included)
    """
    assert protocol in ("easy", "medium", "hard"), \
        f"protocol must be one of 'easy', 'medium', 'hard', got '{protocol}'"

    imlist  = gnd_data["imlist"]   # gallery image filenames (no extension)
    qimlist = gnd_data["qimlist"]  # query image filenames
    gnd     = gnd_data["gnd"]      # list of dicts, one per query

    rows = []
    for qidx, annot in enumerate(gnd):
        query_img = qimlist[qidx]

        # Determine positive gallery indices based on protocol
        if protocol == "easy":
            pos_indices = set(annot.get("easy", []))
        elif protocol == "medium":
            pos_indices = set(annot.get("easy", [])) | set(annot.get("hard", []))
        else:  # hard
            pos_indices = set(annot.get("hard", []))

        junk_indices = set(annot.get("junk", []))

        # Emit all gallery images that have a known label for this query
        labeled_indices = pos_indices | junk_indices
        # For efficiency, only emit positives + junk explicitly;
        # all other (gallery - labeled) are implicit negatives.
        # To keep the file manageable we emit all positive + junk rows,
        # plus a compact negative sample (or all negatives if gallery is small).
        all_gallery = set(range(len(imlist)))
        neg_indices = all_gallery - labeled_indices

        # Positives
        for gidx in pos_indices:
            rows.append({
                "gallery_id":  gidx,
                "query_id":    qidx,
                "gallery_img": imlist[gidx],
                "query_img":   query_img,
                "label":       1,
            })

        # Junk (ambiguous)
        if include_junk:
            for gidx in junk_indices:
                rows.append({
                    "gallery_id":  gidx,
                    "query_id":    qidx,
                    "gallery_img": imlist[gidx],
                    "query_img":   query_img,
                    "label":       -1,
                })

        # Negatives (non-junk, non-positive gallery images)
        for gidx in neg_indices:
            rows.append({
                "gallery_id":  gidx,
                "query_id":    qidx,
                "gallery_img": imlist[gidx],
                "query_img":   query_img,
                "label":       0,
            })

    df = pd.DataFrame(rows, columns=["gallery_id", "query_id", "gallery_img", "query_img", "label"])
    return df


def print_stats(df: pd.DataFrame, protocol: str) -> None:
    """Print summary statistics matching Table 2 in the paper."""
    total    = len(df[df["label"] >= 0])  # exclude junk
    positive = (df["label"] == 1).sum()
    negative = (df["label"] == 0).sum()
    junk     = (df["label"] == -1).sum()

    print(f"\n{'='*55}")
    print(f"  ROxford Ground Truth — Protocol: {protocol.upper()}")
    print(f"{'='*55}")
    print(f"  Gallery images  : {df['gallery_img'].nunique()}")
    print(f"  Query images    : {df['query_img'].nunique()}")
    print(f"  Total pairs     : {total:,}")
    print(f"  Positive pairs  : {positive:,}")
    print(f"  Negative pairs  : {negative:,}")
    print(f"  Junk pairs      : {junk:,} (label=-1, excluded from total)")
    if total > 0:
        selectivity = positive / total
        print(f"  Selectivity     : {selectivity:.2e}")
        print(f"  (Paper Table 2: ~7.8e-05 with revisitop1m distractors; "
              f"without distractors the 5k gallery gives ~1.59e-02.)")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROxford pairwise join ground truth for JoinML benchmarking."
    )
    parser.add_argument(
        "--data_dir", type=str, default=f"{DATA_ROOT}/SemBench/files/roxford5k",
        help="Directory containing (or to download) the ROxford dataset."
    )
    parser.add_argument(
        "--protocol", type=str, default="medium",
        choices=["easy", "medium", "hard"],
        help="Evaluation protocol: easy | medium | hard. Default: medium."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path. Default: <data_dir>/gt_pairwise_<protocol>.csv"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the dataset if not already present."
    )
    parser.add_argument(
        "--skip_images", action="store_true",
        help="If --download is set, skip downloading the large image archive (~1.8 GB)."
    )
    parser.add_argument(
        "--include_junk", action="store_true",
        help="Include junk pairs with label=-1 in the output (excluded by default)."
    )
    parser.add_argument(
        "--emit_tables", action="store_true",
        help="Also emit roxford_left.csv (gallery side, ~5k) and roxford_right.csv "
             "(queries side, 70) plus a positives-only GT file for use by the grid harness."
    )
    parser.add_argument(
        "--scale", type=str, default="5k",
        help="Gallery size target. Keywords: '5k' (default, no distractors, "
             "selectivity ~1.59e-02) | '1m' (1M distractors, selectivity "
             "~7.8e-05, matches JoinML paper Table 2). An integer N overrides: "
             "if N > base 5063 gallery, pad with (N-5063) revisitop1m "
             "distractors; if N <= 5063, truncate the base gallery. "
             "Recommended for 128 GB RAM: 128000 -> 70x128k = ~9M pairs, "
             "~26 GB fused embeddings. Adds ~3 MB manifest download when N > 5063."
    )
    parser.add_argument(
        "--download_r1m_tarballs", type=int, default=0,
        help="Fetch (and extract) this many revisitop1m image tarballs from "
             "http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/. Each "
             "tarball is ~4 GB and holds ~10,000 images. Max=100 covers the "
             "full 1M distractor set (~400 GB). 0 (default) = skip. The grid "
             "runner does NOT need these to exist — missing images become "
             "zero embeddings — but real similarities require them."
    )
    parser.add_argument(
        "--r1m_tarball_start", type=int, default=1,
        help="1-based index of the first tarball to fetch "
             "(used with --download_r1m_tarballs for parallel/resumable pulls)."
    )
    parser.add_argument(
        "--r1m_keep_archive", action="store_true",
        help="Keep the .tar.gz files after extracting. Default: delete to save ~400 GB."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── Download ────────────────────────────────────────────────────────────────
    if args.download:
        download_roxford(data_dir, skip_images=args.skip_images)

    if args.download_r1m_tarballs > 0:
        download_r1m_tarballs(
            data_dir,
            n_tarballs=args.download_r1m_tarballs,
            start=args.r1m_tarball_start,
            extract=True,
            keep_archive=args.r1m_keep_archive,
        )

    # ── Load ground truth ───────────────────────────────────────────────────────
    print(f"Loading ground truth from {data_dir / 'gnd_roxford5k.pkl'} ...")
    gnd_data = load_gnd(data_dir)

    print(f"Gallery size : {len(gnd_data['imlist'])} images")
    print(f"Query size   : {len(gnd_data['qimlist'])} images")

    # ── Build pairwise GT ───────────────────────────────────────────────────────
    print(f"Building pairwise ground truth (protocol={args.protocol}) ...")
    df = build_pairwise_gt(gnd_data, protocol=args.protocol, include_junk=args.include_junk)

    # ── Resolve --scale into a target gallery size ────────────────────────────
    n_base = len(gnd_data["imlist"])
    scale_str = str(args.scale).lower()
    if scale_str == "5k":
        target_gallery = n_base
    elif scale_str == "1m":
        target_gallery = 1_000_000
    else:
        try:
            target_gallery = int(args.scale)
        except ValueError:
            raise SystemExit(
                f"--scale must be '5k', '1m', or an integer; got '{args.scale}'"
            )
        if target_gallery <= 0:
            raise SystemExit(f"--scale must be positive; got {target_gallery}")

    # ── Optional: add revisitop1m distractors to hit the paper's 1M gallery ──
    distractor_names = []
    if target_gallery < n_base:
        raise SystemExit(
            f"--scale {target_gallery} is below the base Roxford5K gallery "
            f"size ({n_base}). Truncating the base gallery would drop GT "
            f"positives and change selectivity. Use a value >= {n_base}."
        )
    if target_gallery > n_base:
        n_dist = target_gallery - n_base
        print(f"Loading {n_dist:,} revisitop1m distractors "
              f"(target gallery = {target_gallery:,}) ...")
        distractor_names = load_r1m_distractors(
            data_dir, n_distractors=n_dist, download=args.download
        )
        n_queries = len(gnd_data["qimlist"])
        dist_rows = pd.DataFrame({
            "gallery_id":  range(n_base, n_base + len(distractor_names)),
            "query_id":    [0] * len(distractor_names),  # placeholder
            "gallery_img": distractor_names,
            "query_img":   ["__DISTRACTOR__"] * len(distractor_names),
            "label":       [0] * len(distractor_names),
        })
        # A distractor is a negative against EVERY query, not just one. We
        # explode each manifest row into n_queries rows with the correct
        # query_id so the cross-product count is exact.
        exploded = []
        for gid, name in enumerate(distractor_names, start=n_base):
            for qidx in range(n_queries):
                exploded.append({
                    "gallery_id":  gid,
                    "query_id":    qidx,
                    "gallery_img": name,
                    "query_img":   gnd_data["qimlist"][qidx],
                    "label":       0,
                })
        if exploded:
            df = pd.concat([df, pd.DataFrame(exploded)], ignore_index=True)
        print(f"After distractor expansion: {len(df):,} pairs "
              f"({n_base + len(distractor_names)} gallery × {n_queries} queries)")

    # ── Stats ───────────────────────────────────────────────────────────────────
    print_stats(df, args.protocol)

    # ── Save ────────────────────────────────────────────────────────────────────
    out_path = args.output or str(data_dir / f"gt_pairwise_{args.protocol}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved ground truth to: {out_path}")
    print(f"Shape: {df.shape}")

    # ── Optional: emit left/right tables and positives-only GT for the grid ──
    if args.emit_tables:
        out_dir = Path(args.output).parent if args.output else (data_dir / "gt")
        out_dir.mkdir(parents=True, exist_ok=True)
        jpg_dir = data_dir / "jpg"
        r1m_jpg_dir = data_dir / "revisitop1m" / "jpg"

        imlist = gnd_data["imlist"]
        qimlist = gnd_data["qimlist"]

        # Left = gallery images (optionally padded with distractors), Right =
        # query images. When --scale 1m, distractor rows point into
        # <data_dir>/revisitop1m/jpg/<name>.jpg (image tarballs must be
        # downloaded separately for grid execution; GT construction only
        # needs the filenames).
        gallery_names = list(imlist) + list(distractor_names)
        # Roxford5K imlist entries are stems (no extension) stored under
        # <data_dir>/jpg/<n>.jpg. The revisitop1m manifest already embeds
        # the .jpg suffix in each line (e.g. '1ee/abc.jpg'), so only append
        # it when missing.
        def _with_jpg(root: Path, n: str) -> str:
            return str(root / (n if n.lower().endswith(".jpg") else f"{n}.jpg"))

        gallery_paths = [_with_jpg(jpg_dir, n) for n in imlist] + \
                        [_with_jpg(r1m_jpg_dir, n) for n in distractor_names]
        left_df = pd.DataFrame({
            "left_id": [f"L{i:06d}" for i in range(len(gallery_names))],
            "left_image": gallery_paths,
            "left_image_name": gallery_names,
        })
        right_df = pd.DataFrame({
            "right_id": [f"R{i:06d}" for i in range(len(qimlist))],
            "right_image": [_with_jpg(jpg_dir, n) for n in qimlist],
            "right_image_name": qimlist,
        })
        pos = df[df["label"] == 1].copy()
        pos["left_id"] = pos["gallery_id"].apply(lambda i: f"L{int(i):06d}")
        pos["right_id"] = pos["query_id"].apply(lambda i: f"R{int(i):06d}")
        pos = pos[["left_id", "right_id", "gallery_img", "query_img", "label"]]

        left_csv = out_dir / "roxford_left.csv"
        right_csv = out_dir / "roxford_right.csv"
        gt_csv = out_dir / "gt_pairwise_roxford.csv"
        left_df.to_csv(left_csv, index=False)
        right_df.to_csv(right_csv, index=False)
        pos.to_csv(gt_csv, index=False)
        print(f"Left table  : {left_csv}   ({len(left_df):,} rows)")
        print(f"Right table : {right_csv}  ({len(right_df):,} rows)")
        print(f"Positive GT : {gt_csv}     ({len(pos):,} rows, label=1 only)")


if __name__ == "__main__":
    main()

