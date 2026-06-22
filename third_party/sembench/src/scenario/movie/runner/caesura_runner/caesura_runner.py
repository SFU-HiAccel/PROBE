"""
Created on August 1, 2025

@author: Jiale Lao

CAESURA runner implementation for movie use case.
"""

import os
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
generic_caesura_path = str(
    Path(__file__).parent.parent.parent.parent.parent
    / "runner"
    / "generic_caesura_runner"
)
sys.path.insert(0, generic_caesura_path)

from runner.generic_caesura_runner.generic_caesura_runner import (
    GenericCaesuraRunner,
)


class CaesuraRunner(GenericCaesuraRunner):
    """CAESURA runner for movie use case."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker: int = 1,
        skip_setup: bool = False,
    ):
        """
        Initialize CAESURA runner for movie case.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
            concurrent_llm_worker: Number of concurrent workers
            skip_setup: Whether to skip setup
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

    def _setup_database_from_files(self):
        """
        Setup CAESURA database using movie data files with required data manipulation.

        Data manipulation requirements:
        - Drop originalScore and scoreSentiment columns
        - Move reviewText column to the last column
        - Drop rows that have null value for reviewText
        """
        from caesura.database.database import Database

        db = Database()

        # Load movie data from MMBench files
        movies_path = str(self.data_path / "Movies_2000.csv")
        reviews_path = str(self.data_path / "Reviews_2000.csv")

        # Load and process reviews data
        reviews_df = pd.read_csv(reviews_path)

        # Data manipulation for reviews:
        # 1. Drop originalScore and scoreSentiment columns if they exist
        columns_to_drop = ["originalScore", "scoreSentiment", "reviewState"]
        for col in columns_to_drop:
            if col in reviews_df.columns:
                reviews_df = reviews_df.drop(columns=[col])

        # 2. Drop rows with null reviewText
        if "reviewText" in reviews_df.columns:
            reviews_df = reviews_df.dropna(subset=["reviewText"])

            # 3. Move reviewText column to the last position
            reviewText_col = reviews_df["reviewText"]
            reviews_df = reviews_df.drop(columns=["reviewText"])
            reviews_df["reviewText"] = reviewText_col

        # Save processed reviews to a temporary file for CAESURA
        temp_reviews_path = str(self.data_path / "temp_reviews_processed.csv")
        reviews_df.to_csv(temp_reviews_path, index=False)

        # Add movies table (no manipulation needed)
        db.add_tabular_table(
            "movies",
            movies_path,
            "a table that contains information about movies including titles, scores, ratings, genres, directors, and other metadata",
        )

        # Add processed reviews table
        db.add_text_table(
            "reviews",
            temp_reviews_path,
            "a table that contains movie reviews with metadata and review text content for sentiment analysis",
        )

        # Link reviews to movies by movie id
        db.link("reviews", "movies", "id")

        # Build relevant values index for key categorical columns
        db.build_relevant_values_index(
            "movies", "genre", "director", "rating", "originalLanguage"
        )

        # Build index for reviews table (excluding dropped columns)
        available_review_cols = []
        for col in ["reviewState", "publicationName", "isTopCritic"]:
            if col in reviews_df.columns:
                available_review_cols.append(col)

        if available_review_cols:
            db.build_relevant_values_index("reviews", *available_review_cols)

        return db

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for each movie query.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with correct columns based on evaluation expectations
        """
        if query_id == 1:
            # Q1: Five clearly positive reviews - returns reviewId
            return pd.DataFrame(columns=["reviewId"])
        elif query_id == 2:
            # Q2: Five positive reviews for "taken_3" - returns reviewId
            return pd.DataFrame(columns=["reviewId"])
        elif query_id == 3:
            # Q3: Count of positive reviews for "taken_3" - returns count
            return pd.DataFrame(columns=["count"])
        elif query_id == 4:
            # Q4: Positivity ratio for "taken_3" - returns ratio/average
            return pd.DataFrame(columns=["positivity_ratio"])
        elif query_id == 5:
            # Q5: Review pairs with same sentiment - returns id, reviewId1, reviewId2
            return pd.DataFrame(columns=["id", "reviewId1", "reviewId2"])
        elif query_id == 6:
            # Q6: Review pairs with opposite sentiment - returns id, reviewId1, reviewId2
            return pd.DataFrame(columns=["id", "reviewId1", "reviewId2"])
        elif query_id == 7:
            # Q7: Sentiment counts - returns scoreSentiment, count
            return pd.DataFrame(columns=["scoreSentiment", "count"])
        else:
            return pd.DataFrame()

    def _execute_q1(self) -> pd.DataFrame:
        """
        Execute Q1: "Five clearly positive reviews (any movie)"

        Returns:
            DataFrame with columns: reviewId
        """
        query_text = self.get_query_text(1, "natural_language")
        results = self.execute_caesura_query(query_text)

        # Extract only reviewId column for evaluation
        if not results.empty and "reviewId" in results.columns:
            return results[["reviewId"]].head(5)
        elif not results.empty:
            # Fallback: use first column if reviewId not found
            first_col = results.columns[0]
            return (
                results[[first_col]]
                .rename(columns={first_col: "reviewId"})
                .head(5)
            )
        else:
            return self._get_empty_results_dataframe(1)

    def _execute_q2(self) -> pd.DataFrame:
        """
        Execute Q2: "Five positive reviews for movie taken_3"

        Returns:
            DataFrame with columns: reviewId
        """
        query_text = self.get_query_text(2, "natural_language")
        results = self.execute_caesura_query(query_text)

        # Extract only reviewId column for evaluation
        if not results.empty and "reviewId" in results.columns:
            return results[["reviewId"]].head(5)
        elif not results.empty:
            # Fallback: use first column if reviewId not found
            first_col = results.columns[0]
            return (
                results[[first_col]]
                .rename(columns={first_col: "reviewId"})
                .head(5)
            )
        else:
            return self._get_empty_results_dataframe(2)

    def _execute_q3(self) -> pd.DataFrame:
        """
        Execute Q3: "Count of positive reviews for movie taken_3"

        Returns:
            DataFrame with columns: count
        """
        query_text = self.get_query_text(3, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For aggregation queries, extract the numeric result
        if not results.empty:
            # Look for count-related columns
            count_cols = [
                col for col in results.columns if "count" in col.lower()
            ]
            if count_cols:
                return results[[count_cols[0]]].rename(
                    columns={count_cols[0]: "count"}
                )
            else:
                # Use first numeric column if available
                numeric_cols = results.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    return results[[numeric_cols[0]]].rename(
                        columns={numeric_cols[0]: "count"}
                    )

        # Return 0 count if no results
        return pd.DataFrame({"count": [0]})

    def _execute_q4(self) -> pd.DataFrame:
        """
        Execute Q4: "Positivity ratio (average of 0/1) for movie taken_3"

        Returns:
            DataFrame with columns: positivity_ratio
        """
        query_text = self.get_query_text(4, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For ratio/average queries, extract the numeric result
        if not results.empty:
            # Look for ratio/average related columns
            ratio_cols = [
                col
                for col in results.columns
                if any(
                    term in col.lower() for term in ["ratio", "average", "avg"]
                )
            ]
            if ratio_cols:
                return results[[ratio_cols[0]]].rename(
                    columns={ratio_cols[0]: "positivity_ratio"}
                )
            else:
                # Use first numeric column if available
                numeric_cols = results.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    return results[[numeric_cols[0]]].rename(
                        columns={numeric_cols[0]: "positivity_ratio"}
                    )

        # Return 0.0 ratio if no results
        return pd.DataFrame({"positivity_ratio": [0.0]})

    def _execute_q5(self) -> pd.DataFrame:
        """
        Execute Q5: "Ten Pairs of reviews that express the *same* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'"

        Returns:
            DataFrame with columns: id, reviewId1, reviewId2
        """
        query_text = self.get_query_text(5, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For pair queries, we need 3 columns: movie id, reviewId1, reviewId2
        if not results.empty and len(results.columns) >= 3:
            # Try to find appropriate columns by name
            id_cols = [
                col
                for col in results.columns
                if "id" in col.lower() and "review" not in col.lower()
            ]
            review_cols = [
                col
                for col in results.columns
                if "review" in col.lower() and "id" in col.lower()
            ]

            if len(id_cols) >= 1 and len(review_cols) >= 2:
                selected_cols = [id_cols[0], review_cols[0], review_cols[1]]
                return (
                    results[selected_cols]
                    .rename(
                        columns={
                            selected_cols[0]: "id",
                            selected_cols[1]: "reviewId1",
                            selected_cols[2]: "reviewId2",
                        }
                    )
                    .head(10)
                )
            else:
                # Fallback: use first 3 columns
                cols = list(results.columns)[:3]
                return (
                    results[cols]
                    .rename(
                        columns={
                            cols[0]: "id",
                            cols[1]: "reviewId1",
                            cols[2]: "reviewId2",
                        }
                    )
                    .head(10)
                )

        return self._get_empty_results_dataframe(5)

    def _execute_q6(self) -> pd.DataFrame:
        """
        Execute Q6: "Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'"

        Returns:
            DataFrame with columns: id, reviewId1, reviewId2
        """
        query_text = self.get_query_text(6, "natural_language")
        results = self.execute_caesura_query(query_text)

        # Same logic as Q5 for pair queries
        if not results.empty and len(results.columns) >= 3:
            # Try to find appropriate columns by name
            id_cols = [
                col
                for col in results.columns
                if "id" in col.lower() and "review" not in col.lower()
            ]
            review_cols = [
                col
                for col in results.columns
                if "review" in col.lower() and "id" in col.lower()
            ]

            if len(id_cols) >= 1 and len(review_cols) >= 2:
                selected_cols = [id_cols[0], review_cols[0], review_cols[1]]
                return (
                    results[selected_cols]
                    .rename(
                        columns={
                            selected_cols[0]: "id",
                            selected_cols[1]: "reviewId1",
                            selected_cols[2]: "reviewId2",
                        }
                    )
                    .head(10)
                )
            else:
                # Fallback: use first 3 columns
                cols = list(results.columns)[:3]
                return (
                    results[cols]
                    .rename(
                        columns={
                            cols[0]: "id",
                            cols[1]: "reviewId1",
                            cols[2]: "reviewId2",
                        }
                    )
                    .head(10)
                )

        return self._get_empty_results_dataframe(6)

    def _execute_q7(self) -> pd.DataFrame:
        """
        Execute Q7: All Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'

        Returns:
            DataFrame with columns: id, reviewId1, reviewId2
        """
        query_text = self.get_query_text(7, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For join queries, we need id and two reviewId columns
        if not results.empty:
            # Look for id and reviewId columns
            id_cols = [
                col
                for col in results.columns
                if col.lower() in ["id", "movieid"]
            ]
            review_cols = [
                col for col in results.columns if "reviewid" in col.lower()
            ]

            if id_cols and len(review_cols) >= 2:
                return results[
                    [id_cols[0], review_cols[0], review_cols[1]]
                ].rename(
                    columns={
                        id_cols[0]: "id",
                        review_cols[0]: "reviewId1",
                        review_cols[1]: "reviewId2",
                    }
                )
            elif len(results.columns) >= 3:
                # Fallback: use first three columns
                cols = list(results.columns)[:3]
                return results[cols].rename(
                    columns={
                        cols[0]: "id",
                        cols[1]: "reviewId1",
                        cols[2]: "reviewId2",
                    }
                )

        return self._get_empty_results_dataframe(7)

    def _execute_q8(self) -> pd.DataFrame:
        """
        Execute Q8: "Calculate the number of positive and negative reviews for movie taken_3"

        Returns:
            DataFrame with columns: scoreSentiment, count
        """
        query_text = self.get_query_text(8, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For group by queries, we need sentiment and count columns
        if not results.empty and len(results.columns) >= 2:
            # Look for sentiment and count columns
            sentiment_cols = [
                col
                for col in results.columns
                if any(term in col.lower() for term in ["sentiment", "score"])
            ]
            count_cols = [
                col for col in results.columns if "count" in col.lower()
            ]

            if sentiment_cols and count_cols:
                return results[[sentiment_cols[0], count_cols[0]]].rename(
                    columns={
                        sentiment_cols[0]: "scoreSentiment",
                        count_cols[0]: "count",
                    }
                )
            elif len(results.columns) >= 2:
                # Fallback: use first two columns
                cols = list(results.columns)[:2]
                return results[cols].rename(
                    columns={cols[0]: "scoreSentiment", cols[1]: "count"}
                )

        return self._get_empty_results_dataframe(8)
