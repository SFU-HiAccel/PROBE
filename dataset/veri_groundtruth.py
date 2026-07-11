"""
VeRi-776 Ground Truth Generator for JoinML-style Pairwise Join Tasks
=====================================================================

Paper context (arXiv:2603.16153, §C.1 & §C.2):
    Query: SELECT AVG(cam2.timestamp - cam1.timestamp)
           FROM surveillance_cam_A AS cam1
           JOIN surveillance_cam_B AS cam2
           ON NL('Two images show the same vehicle: {cam1.image} {cam2.image}')

VeRi-776 (Liu et al. 2016) is a vehicle re-identification dataset: 776 unique
vehicles captured by 20 cameras across a road network. Images are named

    vehicleID_cameraID_sequenceID_timestamp.jpg
    e.g.  0002_c002_00030600_0.jpg
          ^vid ^cam

To repurpose it as a two-table join (per the paper), we split the 20 cameras
into two disjoint sets cam_A and cam_B and take all images on each side. A
pair (img_l, img_r) with img_l in cam_A and img_r in cam_B is a positive
match iff they share the same vehicleID.

Expected statistics (paper Table 2):
    |L| ~= 2.1e4,  |R| ~= 2.8e4,  cross-product ~= 6.0e8
    positive rate ~= 1.4e-3   =>  ~8.4e5 positive pairs

Usage:
    python veri_groundtruth.py --data_dir $LLMSQL_DATA_ROOT/SemBench/files/veri
"""
import os as _os  # noqa: E402  (anonymized path config)
DATA_ROOT = _os.environ.get("LLMSQL_DATA_ROOT", _os.path.expanduser("~"))


import os
import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _parse_filename(fname: str):
    """Parse 'vehicleID_cameraID_sequenceID_timestamp.jpg' into (vid, cam, ts).

    Returns None if the filename does not match the expected format.
    """
    stem = fname.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    vid = parts[0]
    cam = parts[1]
    try:
        ts = int(parts[-1])
    except ValueError:
        ts = 0
    return vid, cam, ts


def _collect_images(data_dir: Path):
    """Walk image_train/, image_test/, image_query/ and build a catalogue.

    Returns a list of dicts: {filepath, vehicle_id, camera_id, timestamp}.
    """
    image_dirs = []
    for sub in ("image_train", "image_test", "image_query"):
        p = data_dir / sub
        if p.is_dir():
            image_dirs.append(p)
    if not image_dirs:
        raise FileNotFoundError(
            f"No image_train/image_test/image_query directory under {data_dir}. "
            "Expected the standard VeRi-776 layout."
        )

    rows = []
    for d in image_dirs:
        for fname in sorted(os.listdir(d)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parsed = _parse_filename(fname)
            if parsed is None:
                continue
            vid, cam, ts = parsed
            rows.append({
                "filepath": str(d / fname),
                "filename": fname,
                "vehicle_id": vid,
                "camera_id": cam,
                "timestamp": ts,
            })
    return rows


def _split_cameras(rows, seed: int):
    """Split the set of camera IDs into cam_A and cam_B such that |A| ~= |B|
    in terms of image count. Uses a deterministic shuffle + greedy balance.
    """
    by_cam = defaultdict(int)
    for r in rows:
        by_cam[r["camera_id"]] += 1
    cams = sorted(by_cam.keys())
    rng = random.Random(seed)
    rng.shuffle(cams)

    total = sum(by_cam.values())
    target = total // 2
    cam_a, cam_b = set(), set()
    acc_a = 0
    for c in cams:
        if acc_a < target:
            cam_a.add(c)
            acc_a += by_cam[c]
        else:
            cam_b.add(c)
    # Fallback in case all went into cam_a
    if not cam_b:
        pivot = cams[len(cams) // 2]
        cam_b.add(pivot)
        cam_a.discard(pivot)
    return cam_a, cam_b


def build_pairwise_gt(data_dir: Path, split_seed: int = 0,
                      scale: int = 0):
    rows = _collect_images(data_dir)
    if not rows:
        raise RuntimeError(f"No VeRi images found under {data_dir}")

    cam_a, cam_b = _split_cameras(rows, seed=split_seed)
    left = [r for r in rows if r["camera_id"] in cam_a]
    right = [r for r in rows if r["camera_id"] in cam_b]

    # Uniform random subsample per side. With positives distributed evenly
    # across vehicle_ids, subsampling preserves selectivity in expectation:
    # E[positives] = (scale/|L_full|) * (scale/|R_full|) * |gt_full|.
    if scale > 0:
        rng = random.Random(split_seed)
        if len(left) > scale:
            left = rng.sample(left, scale)
        if len(right) > scale:
            right = rng.sample(right, scale)

    left_df = pd.DataFrame([
        {
            "left_id": f"L{i:06d}",
            "left_image": r["filepath"],
            "left_vehicle_id": r["vehicle_id"],
            "left_camera_id": r["camera_id"],
            "left_timestamp": r["timestamp"],
        }
        for i, r in enumerate(left)
    ])
    right_df = pd.DataFrame([
        {
            "right_id": f"R{i:06d}",
            "right_image": r["filepath"],
            "right_vehicle_id": r["vehicle_id"],
            "right_camera_id": r["camera_id"],
            "right_timestamp": r["timestamp"],
        }
        for i, r in enumerate(right)
    ])

    # Positives = same vehicle_id across left/right. Do the cross-join on
    # vehicle_id only (this is the ONLY combination that yields label=1).
    pos = left_df.merge(right_df, left_on="left_vehicle_id",
                        right_on="right_vehicle_id", how="inner")
    pos = pos[["left_id", "right_id", "left_image", "right_image",
               "left_vehicle_id", "right_vehicle_id",
               "left_timestamp", "right_timestamp"]]
    pos["label"] = 1
    return left_df, right_df, pos


def print_stats(left_df, right_df, pos_df) -> None:
    total_pairs = len(left_df) * len(right_df)
    n_pos = len(pos_df)
    sel = n_pos / total_pairs if total_pairs else 0.0
    print(f"\n{'='*55}")
    print(f"  VeRi Ground Truth")
    print(f"{'='*55}")
    print(f"  Left  (cam_A) size : {len(left_df):,}")
    print(f"  Right (cam_B) size : {len(right_df):,}")
    print(f"  Cross product      : {total_pairs:,}")
    print(f"  Positive pairs     : {n_pos:,}")
    print(f"  Selectivity        : {sel:.2e}")
    print(f"  (Paper Table 2 reports ~2.1e4 x 2.8e4 = 6.0e8, selectivity 1.4e-3)")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VeRi-776 pairwise join ground truth."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=f"{DATA_ROOT}/SemBench/files/veri",
        help="Directory containing VeRi-776 (with image_train/, image_test/, image_query/)."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for output CSVs. Default: <data_dir>/gt/"
    )
    parser.add_argument(
        "--split_seed", type=int, default=0,
        help="Seed for camera A/B split (deterministic)."
    )
    parser.add_argument(
        "--scale", type=int, default=0,
        help="If > 0, uniformly subsample this many images per side so the "
             "cross product fits memory (paper scale ~2.5k per side = 6e8 "
             "pairs, OOMs at 128 GB). Selectivity is preserved in "
             "expectation. Recommended: 3000 (-> 9M pairs, ~26 GB)."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (data_dir / "gt")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning VeRi images under {data_dir} ...")
    left_df, right_df, pos_df = build_pairwise_gt(
        data_dir, split_seed=args.split_seed, scale=args.scale,
    )
    print_stats(left_df, right_df, pos_df)

    left_csv = out_dir / "veri_left.csv"
    right_csv = out_dir / "veri_right.csv"
    gt_csv = out_dir / "gt_pairwise_veri.csv"

    left_df.to_csv(left_csv, index=False)
    right_df.to_csv(right_csv, index=False)
    pos_df.to_csv(gt_csv, index=False)

    print(f"Left table  : {left_csv}   ({len(left_df):,} rows)")
    print(f"Right table : {right_csv}  ({len(right_df):,} rows)")
    print(f"Positive GT : {gt_csv}     ({len(pos_df):,} rows, label=1 only)")


if __name__ == "__main__":
    main()
