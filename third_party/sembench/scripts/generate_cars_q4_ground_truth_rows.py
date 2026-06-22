"""
Generate CSV files containing ALL ground-truth rows behind Cars Q4
("What is the average age of cars with engine problems?") for one or more
scale factors.

Background
----------
Q4's ground truth (see src/scenario/cars/evaluation/evaluate.py ::
CarsEvaluator._generate_q4_ground_truth) is computed *deterministically* from
the labeled data, NOT from an LLM. A car counts as having an "engine problem"
iff it has a complaint whose ``component_class == "ENGINE"``. The answer is::

    average_age = 2026 - mean(year of those cars)

The shipped ground-truth CSV (files/cars/raw_results/ground_truth/Q4_*.csv)
stores only that single scalar. This script instead emits every *row* that
makes up the answer -- i.e. the rows of ``cars JOIN complaints`` that pass the
engine filter -- so you can inspect them and compute the filter selectivity:

    selectivity = (#rows after engine filter) / (#rows in cars JOIN complaints)

Data layout expected (produced by scenario.cars.preparation.generate_data):
    files/cars/data/full_data/car_data_full.csv               (carries labels)
    files/cars/data/full_data/text_complaints_data_full.csv   (carries labels)
    files/cars/data/sf_<sf>/car_data_<sf>.csv                  (sample; sf != 157376)
    files/cars/data/sf_<sf>/text_complaints_data_<sf>.csv      (sample; sf != 157376)

For sf == 157376 (the full dataset) the full_data tables are used directly,
exactly like the evaluator does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# The full dataset. Smaller scale factors are subsamples of it, so the
# evaluator only subsets when scale_factor != FULL_SCALE_FACTOR.
FULL_SCALE_FACTOR = 157376


def load_labeled_tables(data_root: Path, scale_factor: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the (cars, complaints) tables Q4 needs, with label columns intact.

    Mirrors CarsEvaluator._load_domain_data: read the labeled full_data tables,
    then -- for any non-full scale factor -- restrict them to the IDs present in
    that scale factor's sample files.
    """
    full = data_root / "full_data"
    cars_df = pd.read_csv(full / "car_data_full.csv")
    text_df = pd.read_csv(full / "text_complaints_data_full.csv")

    if scale_factor != FULL_SCALE_FACTOR:
        sample = data_root / f"sf_{scale_factor}"
        cars_sample = pd.read_csv(sample / f"car_data_{scale_factor}.csv")
        text_sample = pd.read_csv(sample / f"text_complaints_data_{scale_factor}.csv")
        cars_df = cars_df[cars_df["car_id"].isin(cars_sample["car_id"])]
        text_df = text_df[text_df["complaint_id"].isin(text_sample["complaint_id"])]

    return cars_df, text_df


def build_q4_ground_truth_rows(
    cars_df: pd.DataFrame, text_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Return (rows, stats) for Q4's ground truth.

    ``rows`` = cars JOIN complaints WHERE component_class == 'ENGINE'. Each row is
    one car together with its engine complaint; ``row['year']`` is what the
    ground truth averages. (In this dataset each car has at most one complaint,
    so this row set matches the GT's distinct-car semantics exactly.)
    """
    engine_complaints = text_df[text_df["component_class"] == "ENGINE"]
    rows = cars_df.merge(engine_complaints, on="car_id", how="inner")

    # Full cars JOIN complaints = the input to the engine filter (denominator).
    full_join_size = len(cars_df.merge(text_df, on="car_id", how="inner"))

    # Put the columns the question cares about first.
    front = [c for c in ("car_id", "year") if c in rows.columns]
    rows = rows[front + [c for c in rows.columns if c not in front]]

    stats = {
        "rows_after_engine_filter": len(rows),  # numerator
        "rows_before_filter_joined": full_join_size,  # denominator
        "selectivity": (len(rows) / full_join_size) if full_join_size else float("nan"),
        "average_age": (2026 - rows["year"].mean()) if len(rows) else float("nan"),
    }
    return rows, stats


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = repo_root / "files" / "cars" / "data"
    default_out_dir = repo_root / "files" / "cars" / "raw_results" / "ground_truth"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scale-factors", type=int, nargs="+", default=[157376, 9836],
        help="Scale factors to generate rows for (default: 157376 9836).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=default_data_root,
        help=f"Path to files/cars/data (default: {default_data_root}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=default_out_dir,
        help=f"Where to write the CSVs (default: {default_out_dir}).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for sf in args.scale_factors:
        try:
            cars_df, text_df = load_labeled_tables(args.data_root, sf)
        except FileNotFoundError as e:
            print(f"[sf={sf}] missing data file: {e.filename}")
            print(
                f"        Run the data prep first, e.g.:\n"
                f"        python -m scenario.cars.preparation.generate_data --scaling_factor {sf}\n"
            )
            continue

        rows, stats = build_q4_ground_truth_rows(cars_df, text_df)
        out_path = args.out_dir / f"Q4_ground_truth_rows_sf{sf}.csv"
        rows.to_csv(out_path, index=False)

        print(f"[sf={sf}] wrote {len(rows)} rows -> {out_path}")
        print(f"        rows after engine filter (numerator)   : {stats['rows_after_engine_filter']}")
        print(f"        rows before filter  cars JOIN complaints: {stats['rows_before_filter_joined']}")
        print(f"        selectivity                            : {stats['selectivity']:.6f}")
        print(f"        average_age (sanity check vs Q4 GT)    : {stats['average_age']:.6f}")


if __name__ == "__main__":
    main()
