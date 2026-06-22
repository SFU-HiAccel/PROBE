"""
Created on Jun 28, 2025

@author: OlgaOvcharenko
"""

from pathlib import Path
from typing import Any, Dict
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluator.generic_evaluator import (
    GenericEvaluator,
    QueryMetricRetrieval,
    QueryMetricAggregation,
)


class MedicalEvaluator(GenericEvaluator):
    """Evaluator for the medical benchmark using the reusable framework."""

    def __init__(self, use_case: str, scale_factor: int = None) -> None:
        super().__init__(use_case, scale_factor)

    def _load_domain_data(self) -> None:
        data_path = self._root / "data"
        self.patient_df = pd.read_csv(data_path / f"patient_data_with_labels{"" if self.scale_factor == 11112 else f"_{int(self.scale_factor)}"}.csv")
        self.audio_df = pd.read_csv(data_path / f"audio_lung_data{"" if self.scale_factor == 11112 else f"_{int(self.scale_factor)}"}.csv")
        self.image_x_ray_df = pd.read_csv(data_path / f"image_x_ray_data{"" if self.scale_factor == 11112 else f"_{int(self.scale_factor)}"}.csv")
        self.symptoms_text_df = pd.read_csv(data_path / f"text_symptoms_data{"" if self.scale_factor == 11112 else f"_{int(self.scale_factor)}"}.csv")
        self.skin_cancer_df = pd.read_csv(data_path / f"image_skin_data{"" if self.scale_factor == 11112 else f"_{int(self.scale_factor)}"}.csv")

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:
        query_name = f"Q{query_id}" if self.scale_factor == 11112 else f"Q{query_id}_{int(self.scale_factor)}"
        gt_path = self._results_path / "ground_truth" / f"{query_name}.csv"
        if gt_path.exists():
            return pd.read_csv(gt_path)

        ground_truth_fn = self._discover_ground_truth_impl(query_id)
        return ground_truth_fn()

    def _generate_q1_ground_truth(self) -> pd.DataFrame:
        sick = self.patient_df[self.patient_df["text_diagnosis"] == "allergy"]
        gt = sick[
            ["age", "gender", "smoking_history", "did_family_have_cancer", "patient_id"]
        ].copy()

        path = self._results_path / "ground_truth" / ("Q1.csv" if self.scale_factor == 11112 else f"Q1_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)
        return gt

    def _generate_q2_ground_truth(self) -> pd.DataFrame:
        healthy = self.patient_df[
            (self.patient_df["smoking_history"] != "Current")
            & (self.patient_df["audio_diagnosis"] == "normal")
        ]
        gt = healthy[
            ["age", "gender", "smoking_history", "did_family_have_cancer", "patient_id"]
        ].copy()

        path = self._results_path / "ground_truth" / ("Q2.csv" if self.scale_factor == 11112 else f"Q2_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)
        return gt

    def _generate_q3_ground_truth(self) -> pd.DataFrame:
        had_cancer = (self.patient_df["did_family_have_cancer"] == 1) & (
            self.patient_df["x_ray_diagnosis"].isin(["00_normal", "none"]) == False
        )
        gt = self.patient_df.loc[had_cancer, "patient_id"]

        path = self._results_path / "ground_truth" / ("Q3.csv" if self.scale_factor == 11112 else f"Q3_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)
        return gt

    def _generate_q4_ground_truth(self) -> pd.DataFrame:
        avg_age = self.patient_df.loc[
            self.patient_df["text_diagnosis"] == "acne", "age"
        ].mean()
        gt = pd.DataFrame({"average_acne_age": [avg_age]})

        path = self._results_path / "ground_truth" / ("Q4.csv" if self.scale_factor == 11112 else f"Q4_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)
        return gt

    def _generate_q5_ground_truth(self) -> pd.DataFrame:
        gt = (
            self.patient_df.loc[
                (self.patient_df["smoking_history"] == "Current")
                &
                (self.patient_df["audio_diagnosis"].isin(["none", "normal"]) == False)
                & 
                (self.patient_df["x_ray_diagnosis"].isin(["none", "00_normal"]) == False),
                "smoking_history",
            ]
            .value_counts()
            .to_frame()
        )
        gt = gt.reset_index()

        path = self._results_path / "ground_truth" / ("Q5.csv" if self.scale_factor == 11112 else f"Q5_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)

        return gt

    def _generate_q6_ground_truth(self) -> pd.DataFrame:
        # num_sick = (
        #     (self.patient_df["text_diagnosis"] != "none").astype(int)
        #     + (
        #         self.patient_df["audio_diagnosis"].isin(["normal", "none"]) == False
        #     ).astype(int)
        #     + (
        #         self.patient_df["x_ray_diagnosis"].isin(["none", "00_normal"]) == False
        #     ).astype(int)
        # )

        condition = ~(
            (self.patient_df["text_diagnosis"] == "none")
            & self.patient_df["audio_diagnosis"].isin(["normal", "none"])
            & self.patient_df["x_ray_diagnosis"].isin(["none", "00_normal"])
        )

        sick_patients = self.patient_df[condition]

        gt = sick_patients[
            (
                self.patient_df["text_diagnosis"]
                + ", "
                + self.patient_df["audio_diagnosis"]
                + ", "
                + self.patient_df["x_ray_diagnosis"]
            ).apply(
                lambda x: (x.split(", ").count("normal") > 0)
                | (x.split(", ").count("00_normal") > 0)
            )
        ]

        # gt = gt.sort_values(by="age", ascending=True).head(1)[
        #     [
        #         "age",
        #         "gender",
        #         "smoking_history",
        #         "did_family_have_cancer",
        #         "patient_id",
        #     ]
        # ]

        path = self._results_path / "ground_truth" / ("Q6.csv" if self.scale_factor == 11112 else f"Q6_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)

        return gt

    def _generate_q7_ground_truth(self) -> pd.DataFrame:
        gt_res = self.patient_df.loc[
            self.patient_df["is_sick"] == 1,
            [
                "age",
                "patient_id",
            ],
        ]

        # avg_age = gt["age"].mean()
        # gt_res = pd.DataFrame({"average_age": [avg_age]})

        path = self._results_path / "ground_truth" / ("Q7.csv" if self.scale_factor == 11112 else f"Q7_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt_res.to_csv(path, index=False)
        
        return gt_res
    
    def _generate_q8_ground_truth(self) -> pd.DataFrame:
        had_cancer = (self.patient_df["did_family_have_cancer"] == 1) & (
            self.patient_df["skin_cancer_diagnosis"] == "malignant"
        )

        gt = self.patient_df.loc[had_cancer, ["patient_id"]]

        path = self._results_path / "ground_truth" / ("Q8.csv" if self.scale_factor == 11112 else f"Q8_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)
        return gt
    
    def _generate_q9_ground_truth(self) -> pd.DataFrame:
        gt = (
            self.patient_df.loc[
                (self.patient_df["skin_cancer_diagnosis"] == "malignant")
                & 
                (self.patient_df["x_ray_diagnosis"].isin(["none", "00_normal"]) == False),
                ["patient_id"]
            ]
        )

        path = self._results_path / "ground_truth" / ("Q9.csv" if self.scale_factor == 11112 else f"Q9_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)
        return gt
    
    def _generate_q10_ground_truth(self) -> pd.DataFrame:
        gt = self.patient_df.join(self.symptoms_text_df.set_index('patient_id'), on='patient_id', how='inner')[['patient_id', 'text_diagnosis']]
        gt["text_diagnosis"] = gt["text_diagnosis"].apply(lambda x: x.lower())
        gt.reset_index(inplace=True)

        path = self._results_path / "ground_truth" / ("Q10.csv" if self.scale_factor == 11112 else f"Q10_{int(self.scale_factor)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(path, index=False)
        
        return gt

    def _evaluate_single_query(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> "QueryMetricRetrieval | QueryMetricAggregation":
        evaluate_fn = self._discover_evaluate_impl(query_id)
        return evaluate_fn(system_results, ground_truth)

    def _evaluate_q1(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        system_results.columns = system_results.columns.str.lower()
        id_column = "patient_id"

        precision = GenericEvaluator.compute_accuracy_score("precision", ground_truth, system_results, id_column=id_column).accuracy
        recall = GenericEvaluator.compute_accuracy_score("recall", ground_truth, system_results, id_column=id_column).accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
        return QueryMetricRetrieval(precision=precision, recall=recall, f1_score=f1)

    def _evaluate_q2(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        system_results.columns = system_results.columns.str.lower()
        id_column = "patient_id"
        
        system_results.drop_duplicates(inplace=True)
        
        precision = GenericEvaluator.compute_accuracy_score("precision", ground_truth, system_results, id_column=id_column).accuracy
        recall = GenericEvaluator.compute_accuracy_score("recall", ground_truth, system_results, id_column=id_column).accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
        return QueryMetricRetrieval(precision=precision, recall=recall, f1_score=f1)
    
    def _evaluate_q3(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        id_column="patient_id"

        system_results.columns = system_results.columns.str.lower()

        print(f"Systems results sample for Q3: {system_results}")
        print(f"Ground truth sample for Q3: {ground_truth}")

        correct_ids_ix = ground_truth[id_column].isin(system_results[id_column].to_list())
        correct_ids = ground_truth.loc[correct_ids_ix,:]
        ground_truth_sample = None
        if correct_ids.empty:
            ground_truth_sample = ground_truth.sample(n=5, random_state=42)

        elif correct_ids.shape[0] > 5:
            raise "Ground truth for Q3 should not contain more than 5 rows. Query contains LIMIT 5."
        
        elif correct_ids.shape[0] < 5:
            ground_truth_sample = pd.concat([correct_ids, ground_truth[correct_ids_ix==False].sample(n=5-correct_ids.shape[0], random_state=42)])
            print(f"Smaller: {ground_truth_sample}")

        elif correct_ids.shape[0] == 5:
            ground_truth_sample = correct_ids
        print(f"Systems results sample for Q3: {system_results}")
        print(f"Ground truth sample for Q3: {ground_truth_sample}")

        id_column = "patient_id"
        
        precision = GenericEvaluator.compute_accuracy_score("precision", ground_truth_sample, system_results, id_column=id_column).accuracy
        recall = GenericEvaluator.compute_accuracy_score("recall", ground_truth_sample, system_results, id_column=id_column).accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
        return QueryMetricRetrieval(precision=precision, recall=recall, f1_score=f1)

    def _evaluate_q4(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        system_results.columns = system_results.columns.str.lower()
        return self._generic_aggregation_evaluation(system_results, ground_truth)

    def _evaluate_q5(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        system_results.columns = system_results.columns.str.lower()
        print(ground_truth)
        return self._generic_aggregation_evaluation(system_results, ground_truth)

    def _evaluate_q6(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        system_results.columns = system_results.columns.str.lower()
        id_column = "patient_id"

        print(system_results)
        print(ground_truth)
        
        precision = GenericEvaluator.compute_accuracy_score("precision", ground_truth, system_results, id_column=id_column).accuracy
        recall = GenericEvaluator.compute_accuracy_score("recall", ground_truth, system_results, id_column=id_column).accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
        return QueryMetricRetrieval(precision=precision, recall=recall, f1_score=f1)
    
    def _evaluate_q7(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        system_results.columns = system_results.columns.str.lower()
        id_column = "patient_id"

        print(system_results)
        print(ground_truth)
        
        precision = GenericEvaluator.compute_accuracy_score("precision", ground_truth, system_results, id_column=id_column).accuracy
        recall = GenericEvaluator.compute_accuracy_score("recall", ground_truth, system_results, id_column=id_column).accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
        return QueryMetricRetrieval(precision=precision, recall=recall, f1_score=f1)
    
    def _evaluate_q8(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        id_column="patient_id"

        system_results.columns = system_results.columns.str.lower()

        print(f"Systems results sample for Q8: {system_results}")
        print(f"Ground truth sample for Q8: {ground_truth}")

        correct_ids_ix = ground_truth[id_column].isin(system_results[id_column].to_list())
        correct_ids = ground_truth.loc[correct_ids_ix,:]
        ground_truth_sample = None
        if correct_ids.empty:
            ground_truth_sample = ground_truth.sample(n=100, random_state=42)

        elif correct_ids.shape[0] > 100:
            raise "Ground truth for Q8 should not contain more than 100 rows. Query contains LIMIT 100."
        
        elif correct_ids.shape[0] < 100:
            ground_truth_sample = pd.concat([correct_ids, ground_truth[correct_ids_ix==False].sample(n=100-correct_ids.shape[0], random_state=42)])
            print(f"Smaller: {ground_truth_sample}")

        elif correct_ids.shape[0] == 100:
            ground_truth_sample = correct_ids
        print(f"Systems results sample for Q8: {system_results}")
        print(f"Ground truth sample for Q8: {ground_truth_sample}")

        precision = GenericEvaluator.compute_accuracy_score("precision", ground_truth_sample, system_results, id_column=id_column).accuracy
        recall = GenericEvaluator.compute_accuracy_score("recall", ground_truth_sample, system_results, id_column=id_column).accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
        return QueryMetricRetrieval(precision=precision, recall=recall, f1_score=f1)
    
    def _evaluate_q9(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        id_column="patient_id"
        system_results.columns = system_results.columns.str.lower()
        print(system_results)
        print(ground_truth)

        precision = GenericEvaluator.compute_accuracy_score("precision", ground_truth, system_results, id_column=id_column).accuracy
        recall = GenericEvaluator.compute_accuracy_score("recall", ground_truth, system_results, id_column=id_column).accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
        return QueryMetricRetrieval(precision=precision, recall=recall, f1_score=f1)
    
    def _evaluate_q10(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        id_column = "patient_id"
        result_column = "text_diagnosis"

        system_results.columns = system_results.columns.str.lower()
        system_results["text_diagnosis"] = system_results["text_diagnosis"].apply(lambda x: str(x).lower().replace("\n", ""))

        gt = ground_truth.sort_values(id_column)[result_column]
        query = system_results.sort_values(id_column)[result_column]
        
        (precision, recall, f1, _) = precision_recall_fscore_support(gt, query, average="macro")
        return QueryMetricRetrieval(precision=precision, recall=recall, f1_score=f1)
        
        # return GenericEvaluator.compute_accuracy_score("f1-score-classify", ground_truth, system_results, id_column="patient_id", result_column="text_diagnosis")
