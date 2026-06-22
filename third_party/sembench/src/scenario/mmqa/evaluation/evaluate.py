"""Evaluator for the MMQA dataset."""

import json
from typing import Union

import pandas as pd

from evaluator.generic_evaluator import (
    GenericEvaluator,
    QueryMetricAggregation,
    QueryMetricRetrieval,
    QueryMetricRank,
)


def compute_metrics(results: list, ground_truth: Union[set, list]):
    tp = 0
    fp = 0

    for item in results:
        if item in ground_truth:
            tp += 1
        else:
            fp += 1

    assert tp + fp == len(
        results
    ), "True Positives and False Positives do not match the results length."
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(ground_truth) if len(ground_truth) > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return QueryMetricRetrieval(precision, recall, f1_score)


class MMQAEvaluator(GenericEvaluator):
    def __init__(self, use_case: str, scale_factor: int) -> None:
        super().__init__(use_case, scale_factor)

        self.use_case = use_case

    def _load_domain_data(self) -> None:
        pass

    def _get_ground_truth(self, query_id: int) -> str:
        return self._root / "query" / "natural_language" / f"q{query_id}.json"

    def _evaluate_single_query(
        self, query_id: int, system_results: pd.DataFrame, ground_truth: str
    ) -> "QueryMetricRetrieval | QueryMetricAggregation | QueryMetricRank":
        """Evaluate a single query based on its type."""
        try:
            query_id = int(query_id)
        except ValueError:
            query_id = int(query_id[:-1])  # "3a" -> "3"
        evaluate_fn = self._discover_evaluate_impl(query_id)
        return evaluate_fn(system_results, ground_truth)

    def _evaluate_q1(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = []
        for _, row in system_results.iterrows():
            results.append(row["director"].strip(' "').lower())

        with open(ground_truth_filepath, "r") as f:
            ground_truth = {
                g.strip().lower() for g in json.load(f).get("ground_truth")
            }

        return compute_metrics(results, ground_truth)

    def _evaluate_q2(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        if "uri" in system_results.columns:  # for BigQuery
            system_results.rename(columns={"uri": "image_id"}, inplace=True)
        if "filename" in system_results.columns:  # for Palimpzest
            system_results.rename(
                columns={"filename": "image_id"}, inplace=True
            )

        results = set()
        for _, row in system_results.iterrows():
            image_id = row["image_id"].split("/")[-1]
            image_id = image_id.replace("%2e", ".")

            if len(row) == 2:
                results.add((row["ID"], image_id))
            elif len(row) == 3:
                results.add(
                    (
                        row["ID"],
                        image_id,
                        str(row["color"]).strip().lower(),
                    )
                )
            else:
                raise ValueError(
                    f"Unexpected number of columns: {len(row)} in the results."
                )

        with open(ground_truth_filepath, "r") as f:
            ground_truth = json.load(f).get("ground_truth")
            ground_truth = set(tuple(g) for g in ground_truth)

        return compute_metrics(results, ground_truth)

    def _evaluate_q3(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = system_results["title"].tolist()

        with open(ground_truth_filepath, "r") as f:
            ground_truth = set(json.load(f).get("ground_truth"))

        return compute_metrics(results, ground_truth)

    def _evaluate_q4(
        self, system_results: pd.DataFrame, ground_truth_filepath: str) -> QueryMetricRetrieval:
        if system_results.empty or "genre" not in system_results.columns:
            return compute_metrics([], set())  # or whatever empty result is appropriate
        results = []
        for _, row in system_results.iterrows():
            genre = row["genre"]
            
            # Skip rows with NaN or non-string genre values
            if pd.isna(genre):
                continue
            
            genre = str(genre).strip().lower()
            
            movies_in_genre = row["movies_in_genre"]
            if pd.isna(movies_in_genre):
                continue
                
            for movie in str(movies_in_genre).split(","):
                results.append((genre, movie.strip().lower()))

        with open(ground_truth_filepath, "r") as f:
            raw_ground_truth = json.load(f).get("ground_truth")

        ground_truth = set()
        for genre, movies in raw_ground_truth.items():
            for m in movies:
                ground_truth.add((genre.strip().lower(), m.strip().lower()))

        return compute_metrics(results, ground_truth)

    def _evaluate_q5(
    self, system_results: pd.DataFrame, ground_truth_filepath: str) -> QueryMetricRetrieval:
        results = []
        for _, row in system_results.iterrows():
            if "_output" in row:
                value = row["_output"]
            elif "actor" in row:
                value = row["actor"]
            else:
                raise ValueError(
                    "Expected either '_output' or 'actor' column in the results."
                )
            
            # Skip NaN/None values or non-string values
            if pd.isna(value) or not isinstance(value, str):
                continue
                
            results.append(value.strip().lower())

        with open(ground_truth_filepath, "r") as f:
            ground_truth = set(json.load(f).get("ground_truth"))
            ground_truth = {g.strip().lower() for g in ground_truth}

        return compute_metrics(results, ground_truth)

    def _evaluate_q6(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        print(system_results.columns)
        
        # Handle empty DataFrame
        if system_results.empty or "Airlines" not in system_results.columns:
            results = []
        else:
            results = system_results["Airlines"].tolist()

        with open(ground_truth_filepath, "r") as f:
            ground_truth = set(json.load(f).get("ground_truth"))

        return compute_metrics(results, ground_truth)

    def _evaluate_q7(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        if "uri" in system_results.columns:  # for BigQuery
            system_results.rename(columns={"uri": "image_id"}, inplace=True)
        if "filename" in system_results.columns:  # for Palimpzest
            system_results.rename(
                columns={"filename": "image_id"}, inplace=True
            )

        results = set()
        for _, row in system_results.iterrows():
            image_id = row["image_id"].split("/")[-1]
            image_id = image_id.replace("%2e", ".")
            results.add((row["Airlines"], image_id))

        with open(ground_truth_filepath, "r") as f:
            ground_truth = json.load(f).get("ground_truth", [])
            ground_truth = set(tuple(g) for g in ground_truth)

        return compute_metrics(results, ground_truth)
