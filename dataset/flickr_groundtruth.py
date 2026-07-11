"""
Flickr30K Ground Truth Generator for JoinML-style Pairwise Join Tasks
=====================================================================

Paper context (arXiv:2603.16153, §C.2):
    Query: SELECT COUNT(*)
           FROM flickr_images AS img
           JOIN flickr_captions AS cap
           ON NL('Caption {cap.text} describes the image {img.image}')

Flickr30K (Young et al. 2014) contains ~31,783 Flickr images, each annotated
with exactly 5 human-written captions, for ~158,915 (image, caption) pairs.

To repurpose as a two-table join we build:
    left  = images   (~3.2e4 rows)
    right = captions (~1.6e5 rows, 5 per image)
A pair is positive iff the caption was one of the 5 captions written for that
image (i.e., the original human annotation).

Expected statistics (paper Table 2):
    |L| ~= 3.2e4,  |R| ~= 1.6e5,  cross-product ~= 5.1e9
    positive rate ~= 3.1e-5   =>  ~1.6e5 positive pairs

Usage:
    # If you already have results_20130124.token and flickr30k-images/:
    python flickr_groundtruth.py --data_dir /path/to/flickr30k

    # Auto-download (images + annotations) via kaggle or HuggingFace:
    python flickr_groundtruth.py --data_dir /path/to/flickr30k --download
"""
import os as _os  # noqa: E402  (anonymized path config)
DATA_ROOT = _os.environ.get("LLMSQL_DATA_ROOT", _os.path.expanduser("~"))


import os
import argparse
import ast
import json
import zipfile
from pathlib import Path

import pandas as pd


# Canonical annotation filenames. Covers three distribution formats:
#   results_20130124.token     -> original Young et al. 2014 release
#   flickr_annotations_30k.csv -> HuggingFace nlphuji/flickr30k release
#   results.csv / captions.txt -> older community CSV mirrors
ANN_CANDIDATES = [
    "flickr_annotations_30k.csv",
    "results_20130124.token",
    "results.csv",
    "captions.txt",
]


def _find_annotations(data_dir: Path) -> Path:
    for name in ANN_CANDIDATES:
        p = data_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No Flickr30K annotation file found under {data_dir}.\n"
        f"Looked for: {ANN_CANDIDATES}\n"
        "Either:\n"
        "  1. run this script with --download to fetch nlphuji/flickr30k from HuggingFace, or\n"
        "  2. manually place flickr_annotations_30k.csv (or results_20130124.token) in --data_dir."
    )


def _find_image_dir(data_dir: Path) -> Path:
    # Auto-extract flickr30k-images.zip the first time we see it (HF release).
    zip_path = data_dir / "flickr30k-images.zip"
    if zip_path.is_file() and not (data_dir / "flickr30k-images").is_dir():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
    for name in ("flickr30k-images", "flickr30k_images", "images"):
        p = data_dir / name
        if p.is_dir() and any(p.iterdir()):
            return p
    raise FileNotFoundError(
        f"No image directory found under {data_dir}. "
        "Expected flickr30k-images/ or flickr30k_images/ or images/ "
        "(or flickr30k-images.zip to auto-extract)."
    )


def _parse_caption_list(val) -> list:
    """The HuggingFace nlphuji/flickr30k CSV encodes the 5-caption list as a
    Python-literal string, e.g. "[\'A man...\', \'Another...\', ...]" — but
    also sometimes as a JSON array. Try both.
    """
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if not s or s in ("nan", "None"):
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, (list, tuple)):
            return list(out)
    except (ValueError, SyntaxError):
        pass
    try:
        out = json.loads(s)
        if isinstance(out, list):
            return out
    except json.JSONDecodeError:
        pass
    return [s]


def _parse_annotations(ann_path: Path) -> pd.DataFrame:
    """Parse three supported formats:

    1. Tokenised .token (Young et al. original), one caption per line:
           <image_filename>#<caption_idx>\t<caption_text>

    2. HuggingFace nlphuji/flickr30k CSV:
           columns include 'filename' and 'caption' where caption holds a
           5-element list encoded as Python-literal or JSON string.

    3. Flat CSV mirrors: (image_name, comment_number, comment).
    """
    if ann_path.suffix == ".token" or ann_path.name.endswith(".token"):
        rows = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                key, text = line.split("\t", 1)
                if "#" in key:
                    img_name, cap_idx = key.rsplit("#", 1)
                else:
                    img_name, cap_idx = key, "0"
                rows.append({"image_name": img_name,
                             "caption_idx": cap_idx,
                             "caption": text})
        return pd.DataFrame(rows)

    df = pd.read_csv(ann_path, sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}
    img_col = cols.get("filename") or cols.get("image_name") or cols.get("image")
    # 'raw' is the caption-list column used by nlphuji/flickr30k on HuggingFace.
    cap_col = (cols.get("raw") or cols.get("caption") or cols.get("captions")
               or cols.get("comment") or cols.get("text") or cols.get("sentences"))
    if img_col is None or cap_col is None:
        raise ValueError(
            f"Unable to locate image/caption columns in {ann_path}: {df.columns.tolist()}"
        )

    # Detect HuggingFace list-per-row format by probing the first non-null value.
    probe = df[cap_col].dropna().iloc[0] if len(df[cap_col].dropna()) else ""
    is_list_form = isinstance(probe, (list, tuple)) or (
        isinstance(probe, str) and probe.strip().startswith(("[", "("))
    )

    rows = []
    if is_list_form:
        for _, r in df.iterrows():
            caps = _parse_caption_list(r[cap_col])
            for i, cap in enumerate(caps):
                rows.append({"image_name": str(r[img_col]),
                             "caption_idx": str(i),
                             "caption": cap})
    else:
        idx_col = cols.get("comment_number") or cols.get("caption_idx")
        for _, r in df.iterrows():
            rows.append({"image_name": str(r[img_col]),
                         "caption_idx": str(r[idx_col]) if idx_col else "0",
                         "caption": str(r[cap_col])})
    return pd.DataFrame(rows)


_HF_BASE = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main"


def _http_download(url: str, dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _hook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            print(f"\r  {dest.name}: {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def _download(data_dir: Path) -> None:
    """Fetch the two HF release artefacts directly by URL. Works without the
    huggingface_hub package; falls back to snapshot_download if the raw URLs
    are unreachable.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "flickr_annotations_30k.csv"
    zip_path = data_dir / "flickr30k-images.zip"

    try:
        if not csv_path.exists():
            print(f"Downloading {csv_path.name} ...")
            _http_download(f"{_HF_BASE}/flickr_annotations_30k.csv", csv_path)
        if not zip_path.exists():
            print(f"Downloading {zip_path.name} (~4.4 GB) ...")
            _http_download(f"{_HF_BASE}/flickr30k-images.zip", zip_path)
        return
    except Exception as e:
        print(f"Direct HTTPS download failed ({e}); trying huggingface_hub.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "Direct HTTPS download failed and huggingface_hub is not installed. "
            "Either `pip install huggingface_hub` or manually place "
            "flickr_annotations_30k.csv + flickr30k-images.zip under --data_dir."
        )
    print("Falling back to snapshot_download nlphuji/flickr30k ...")
    snapshot_download(repo_id="nlphuji/flickr30k", repo_type="dataset",
                      local_dir=str(data_dir), local_dir_use_symlinks=False)


def build_pairwise_gt(data_dir: Path, scale: int = 0, seed: int = 0):
    ann_path = _find_annotations(data_dir)
    img_dir = _find_image_dir(data_dir)
    print(f"Annotations : {ann_path}")
    print(f"Image dir   : {img_dir}")

    ann_df = _parse_annotations(ann_path)

    # Build left table (unique images)
    img_names = ann_df["image_name"].drop_duplicates().tolist()

    # Build right table (all captions)
    right_rows_full = []
    for i, r in enumerate(ann_df.itertuples(index=False)):
        right_rows_full.append({
            "right_id_full": i,
            "right_caption": r.caption,
            "right_image_name": r.image_name,
            "right_caption_idx": r.caption_idx,
        })

    # Uniform independent subsample per side. Because each image in the
    # full release has exactly 5 positive captions drawn from |R_full|
    # captions, the per-pair positive probability is 5/|R_full| regardless
    # of sample size — so expected selectivity is invariant:
    # E[selectivity] = 5/|R_full| ~= 3.1e-5.
    if scale > 0:
        import random as _rnd
        rng = _rnd.Random(seed)
        if len(img_names) > scale:
            img_names = rng.sample(img_names, scale)
        if len(right_rows_full) > scale:
            right_rows_full = rng.sample(right_rows_full, scale)

    left_rows = []
    for i, name in enumerate(img_names):
        left_rows.append({
            "left_id": f"L{i:06d}",
            "left_image": str(img_dir / name),
            "left_image_name": name,
        })
    left_df = pd.DataFrame(left_rows)

    right_rows = []
    for i, r in enumerate(right_rows_full):
        right_rows.append({
            "right_id": f"R{i:06d}",
            "right_caption": r["right_caption"],
            "right_image_name": r["right_image_name"],
            "right_caption_idx": r["right_caption_idx"],
        })
    right_df = pd.DataFrame(right_rows)

    # Positives: right_image_name == left_image_name (one row per caption-image annotation)
    pos = left_df.merge(
        right_df, left_on="left_image_name", right_on="right_image_name", how="inner"
    )
    pos = pos[["left_id", "right_id", "left_image", "right_caption",
               "left_image_name", "right_image_name", "right_caption_idx"]]
    pos["label"] = 1
    return left_df, right_df, pos


def print_stats(left_df, right_df, pos_df) -> None:
    total_pairs = len(left_df) * len(right_df)
    n_pos = len(pos_df)
    sel = n_pos / total_pairs if total_pairs else 0.0
    print(f"\n{'='*55}")
    print(f"  Flickr30K Ground Truth")
    print(f"{'='*55}")
    print(f"  Left  (images)   : {len(left_df):,}")
    print(f"  Right (captions) : {len(right_df):,}")
    print(f"  Cross product    : {total_pairs:,}")
    print(f"  Positive pairs   : {n_pos:,}")
    print(f"  Selectivity      : {sel:.2e}")
    print(f"  (Paper Table 2 reports ~3.2e4 x 1.6e5 = 5.1e9, selectivity 3.1e-5)")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Flickr30K pairwise join ground truth."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=f"{DATA_ROOT}/SemBench/files/flickr30k",
        help="Directory containing Flickr30K (results_20130124.token + flickr30k-images/)."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for output CSVs. Default: <data_dir>/gt/"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Auto-download Flickr30K (via HuggingFace nlphuji/flickr30k)."
    )
    parser.add_argument(
        "--scale", type=int, default=0,
        help="If > 0, uniformly subsample this many images and this many "
             "captions (independent). Selectivity stays ~3.1e-5 because "
             "per-pair positive probability is 5/|R_full|. Paper scale "
             "(|L|=3.2e4, |R|=1.6e5 -> 5.1e9 pairs) OOMs at 128 GB. "
             "Recommended: 3000 -> 9M pairs, ~26 GB."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the --scale subsample."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.download:
        _download(data_dir)

    out_dir = Path(args.output_dir) if args.output_dir else (data_dir / "gt")
    out_dir.mkdir(parents=True, exist_ok=True)

    left_df, right_df, pos_df = build_pairwise_gt(
        data_dir, scale=args.scale, seed=args.seed,
    )
    print_stats(left_df, right_df, pos_df)

    left_csv = out_dir / "flickr_left.csv"
    right_csv = out_dir / "flickr_right.csv"
    gt_csv = out_dir / "gt_pairwise_flickr.csv"

    left_df.to_csv(left_csv, index=False)
    right_df.to_csv(right_csv, index=False)
    pos_df.to_csv(gt_csv, index=False)

    print(f"Left table  : {left_csv}   ({len(left_df):,} rows)")
    print(f"Right table : {right_csv}  ({len(right_df):,} rows)")
    print(f"Positive GT : {gt_csv}     ({len(pos_df):,} rows, label=1 only)")


if __name__ == "__main__":
    main()
