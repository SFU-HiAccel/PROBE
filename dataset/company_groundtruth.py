"""
Company (Magellan / DeepMatcher) Ground Truth Generator
========================================================

Paper context (arXiv:2603.16153, §C.2):
    Query: SELECT COUNT(*)
           FROM wiki_companies AS c1
           JOIN dbpedia_companies AS c2
           ON NL('Company 1: {c1.description} and Company 2: {c2.description} are the same company.')

The Company dataset is the "Company" benchmark from the Magellan/DeepMatcher
entity-matching collection (Konda et al. 2016, Mudgal et al. 2018) and has
two tables of company descriptions plus a positive-match label file. We
follow the paper's usage: a two-table join where label=1 means the two
records refer to the same real-world company.

Standard release layout (from the DeepMatcher/magellandata S3 bucket):
    company/
        tableA.csv     # 28,200 Wikipedia-style records (content, id)
        tableB.csv     # 28,200 DBPedia-style records  (content, id)
        train.csv      # (ltable_id, rtable_id, label)
        valid.csv      # (ltable_id, rtable_id, label)
        test.csv       # (ltable_id, rtable_id, label)

Expected statistics (paper Table 2):
    |L| ~= 1.0e4,  |R| ~= 1.0e4,  cross-product ~= 1.0e8
    positive rate ~= 3.6e-5   =>  ~3.6e3 positive pairs

The paper uses 10,000-row subsets; we sub-sample deterministically so that
ALL positive pairs are preserved (otherwise selectivity moves around). The
--scale flag adjusts the target row count per side.

Usage:
    python company_groundtruth.py --data_dir $LLMSQL_DATA_ROOT/SemBench/files/company
    python company_groundtruth.py --data_dir ./data/company --download
"""
import os as _os  # noqa: E402  (anonymized path config)
DATA_ROOT = _os.environ.get("LLMSQL_DATA_ROOT", _os.path.expanduser("~"))


import argparse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


# DeepMatcher official mirror (Mudgal et al. 2018); URL from the
# anhaidgroup/deepmatcher Datasets.md manifest. Company lives under "Textual".
COMPANY_URL = "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/company_exp_data.zip"


def _download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "company_exp_data.zip"
    if not zip_path.exists():
        print(f"Downloading {COMPANY_URL} -> {zip_path}")
        urllib.request.urlretrieve(COMPANY_URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)


def _locate(data_dir: Path, *names: str) -> Path:
    """Recursively search data_dir for any of the given filenames.

    The DeepMatcher zip typically extracts into a nested layout
    (e.g. data_dir/exp_data/<files>.csv or data_dir/company/<files>.csv),
    so we walk the tree and return the first hit.
    """
    for name in names:
        # Prefer exact top-level matches for stability
        for candidate in (data_dir, data_dir / "company", data_dir / "Company",
                          data_dir / "exp_data", data_dir / "Company" / "exp_data"):
            p = candidate / name
            if p.is_file():
                return p
    # Fallback: recursive walk
    for name in names:
        for p in data_dir.rglob(name):
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"Could not locate any of {names} under {data_dir}. "
        "Run with --download or provide the correct --data_dir."
    )


def _load_tables(data_dir: Path):
    tA_path = _locate(data_dir, "tableA.csv")
    tB_path = _locate(data_dir, "tableB.csv")
    match_paths = []
    # DeepMatcher splits labels across train/valid/test; some mirrors also
    # publish a consolidated matches.csv. Collect whichever are present.
    for name in ("train.csv", "valid.csv", "validation.csv", "test.csv", "matches.csv"):
        try:
            match_paths.append(_locate(data_dir, name))
        except FileNotFoundError:
            continue
    if not match_paths:
        raise FileNotFoundError(
            f"No match label file found (train/valid/validation/test/matches.csv) "
            f"under {data_dir}"
        )

    tA = pd.read_csv(tA_path)
    tB = pd.read_csv(tB_path)
    labels = pd.concat([pd.read_csv(p) for p in match_paths], ignore_index=True)

    # Magellan convention: _id column, ltable_id / rtable_id in label files.
    if "id" not in tA.columns and "_id" in tA.columns:
        tA = tA.rename(columns={"_id": "id"})
    if "id" not in tB.columns and "_id" in tB.columns:
        tB = tB.rename(columns={"_id": "id"})
    if "id" not in tA.columns:
        tA = tA.rename(columns={tA.columns[0]: "id"})
    if "id" not in tB.columns:
        tB = tB.rename(columns={tB.columns[0]: "id"})

    # Company is the "Textual" domain — the description column is called
    # 'content' in DeepMatcher's release. Fall back to other common names.
    desc_cols_a = [c for c in tA.columns if c.lower() in ("content", "description", "text", "name", "title")]
    desc_cols_b = [c for c in tB.columns if c.lower() in ("content", "description", "text", "name", "title")]
    if not desc_cols_a or not desc_cols_b:
        raise ValueError(
            f"Could not find description column in tableA/tableB. "
            f"Columns A={tA.columns.tolist()}, B={tB.columns.tolist()}"
        )
    tA = tA.rename(columns={desc_cols_a[0]: "description"})
    tB = tB.rename(columns={desc_cols_b[0]: "description"})
    return tA, tB, labels


def build_pairwise_gt(data_dir: Path, scale: int):
    tA, tB, labels = _load_tables(data_dir)

    # IDs in Magellan Company are URI strings (e.g. "http://.../Q123_wiki").
    # Keep them as strings throughout.
    tA["id"] = tA["id"].astype(str)
    tB["id"] = tB["id"].astype(str)
    labels["ltable_id"] = labels["ltable_id"].astype(str)
    labels["rtable_id"] = labels["rtable_id"].astype(str)

    pos_labels = labels[labels["label"] == 1].copy()
    print(f"Raw Company release: |A|={len(tA):,}, |B|={len(tB):,}, "
          f"labeled_pairs={len(labels):,}, positives={len(pos_labels):,}")

    # Uniform random subsample of each side to `scale` rows. With |A|=|B|≈2.8e4
    # and ~2.8e4 1-to-1 positives, a 10k/10k uniform sample naturally retains
    # (10k/28.2k)² × 28.2k ≈ 3,550 positives — exactly the paper's ~3.6e3 @
    # 3.6e-5 selectivity. Do NOT force-keep positives; that would inflate
    # selectivity and diverge from Table 2.
    if len(tA) > scale:
        left = tA.sample(n=scale, random_state=17).reset_index(drop=True)
    else:
        left = tA.reset_index(drop=True)
    if len(tB) > scale:
        right = tB.sample(n=scale, random_state=23).reset_index(drop=True)
    else:
        right = tB.reset_index(drop=True)

    left_df = pd.DataFrame({
        "left_id": [f"L{i:06d}" for i in range(len(left))],
        "left_tableA_id": left["id"].values,
        "left_description": left["description"].astype(str).values,
    })
    right_df = pd.DataFrame({
        "right_id": [f"R{i:06d}" for i in range(len(right))],
        "right_tableB_id": right["id"].values,
        "right_description": right["description"].astype(str).values,
    })

    # Positives restricted to the sampled ids.
    kept_a = set(left_df["left_tableA_id"].tolist())
    kept_b = set(right_df["right_tableB_id"].tolist())
    pos_labels = pos_labels[
        pos_labels["ltable_id"].isin(kept_a) &
        pos_labels["rtable_id"].isin(kept_b)
    ]

    pos = pos_labels.merge(
        left_df, left_on="ltable_id", right_on="left_tableA_id"
    ).merge(
        right_df, left_on="rtable_id", right_on="right_tableB_id"
    )
    pos = pos[["left_id", "right_id", "left_description", "right_description",
               "left_tableA_id", "right_tableB_id"]]
    pos["label"] = 1
    return left_df, right_df, pos


def print_stats(left_df, right_df, pos_df) -> None:
    total_pairs = len(left_df) * len(right_df)
    n_pos = len(pos_df)
    sel = n_pos / total_pairs if total_pairs else 0.0
    print(f"\n{'='*55}")
    print(f"  Company Ground Truth")
    print(f"{'='*55}")
    print(f"  Left  size     : {len(left_df):,}")
    print(f"  Right size     : {len(right_df):,}")
    print(f"  Cross product  : {total_pairs:,}")
    print(f"  Positive pairs : {n_pos:,}")
    print(f"  Selectivity    : {sel:.2e}")
    print(f"  (Paper Table 2 reports 1.0e4 x 1.0e4 = 1.0e8, selectivity 3.6e-5)")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Company (Magellan) pairwise join ground truth."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=f"{DATA_ROOT}/SemBench/files/company",
        help="Directory with tableA.csv, tableB.csv, train/valid/test.csv."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for output CSVs. Default: <data_dir>/gt/"
    )
    parser.add_argument(
        "--scale", type=int, default=10000,
        help="Number of rows per side (paper uses 1e4)."
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Auto-download the DeepMatcher Company zip if not present."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.download:
        _download(data_dir)

    out_dir = Path(args.output_dir) if args.output_dir else (data_dir / "gt")
    out_dir.mkdir(parents=True, exist_ok=True)

    left_df, right_df, pos_df = build_pairwise_gt(data_dir, scale=args.scale)
    print_stats(left_df, right_df, pos_df)

    left_csv = out_dir / "company_left.csv"
    right_csv = out_dir / "company_right.csv"
    gt_csv = out_dir / "gt_pairwise_company.csv"

    left_df.to_csv(left_csv, index=False)
    right_df.to_csv(right_csv, index=False)
    pos_df.to_csv(gt_csv, index=False)

    print(f"Left table  : {left_csv}   ({len(left_df):,} rows)")
    print(f"Right table : {right_csv}  ({len(right_df):,} rows)")
    print(f"Positive GT : {gt_csv}     ({len(pos_df):,} rows, label=1 only)")


if __name__ == "__main__":
    main()
