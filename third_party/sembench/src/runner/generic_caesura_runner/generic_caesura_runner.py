"""
Created on August 1, 2025

@author: Jiale Lao

Generic CAESURA runner for MMBench-System
"""

import time
import pandas as pd
from typing import Dict, Any, List, Optional
import sys
import os
import re

# Set up sys.path to ensure consistent imports for local CAESURA code
_caesura_path = os.path.join(os.path.dirname(__file__))
if _caesura_path not in sys.path:
    sys.path.insert(0, _caesura_path)

# Import CAESURA components
from caesura.main import Caesura
from caesura.scenarios import get_database

import traceback
from ..generic_runner import GenericRunner, GenericQueryMetric


class GenericCaesuraRunner(GenericRunner):
    """Generic CAESURA runner for MMBench-System."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str,
        concurrent_llm_worker: int,
        skip_setup: bool = False,
    ):
        """
        Initialize the CAESURA runner.

        Args:
            use_case: The use case to run (e.g., 'movie', 'detective', 'animals')
            model_name: Name of the model to use (e.g., 'gpt-4-0613', 'gpt-3.5-turbo-0613')
            concurrent_llm_worker: Number of concurrent LLM workers (not used by CAESURA)
            skip_setup: Whether to skip setup (inherited from GenericRunner)
        """
        super().__init__(
            use_case,
            model_name,
            scale_factor,
            concurrent_llm_worker,
            skip_setup,
        )

        # Map common model names to CAESURA format
        self.caesura_model = self._map_model_name(model_name)

        # Initialize CAESURA database using MMBench data files
        try:
            self.database = self._setup_database_from_files()
        except Exception as e:
            raise ValueError(
                f"Failed to setup CAESURA database for use case '{use_case}': {e}"
            ) from e

        # Initialize single CAESURA agent for reuse
        self.caesura_agent = Caesura(
            database=self.database,
            model_name=self.caesura_model,
            interactive=False,
        )

    def _map_model_name(self, model_name: str) -> str:
        """Map MMBench model names to CAESURA model names."""
        model_mapping = {
            "gpt-4": "gpt-4-0613",
            "gpt-3.5": "gpt-3.5-turbo-0613",
        }
        return model_mapping.get(model_name, "gpt-4-0613")  # Default to GPT-4

    def _setup_database_from_files(self):
        """Setup CAESURA database using MMBench data files."""
        from caesura.database.database import Database

        db = Database()

        if self.use_case == "movie":
            # Load movie data from MMBench files
            movies_path = str(self.data_path / "Movies_2000.csv")
            reviews_path = str(self.data_path / "Reviews_2000.csv")

            # Add movies table
            db.add_tabular_table(
                "movies",
                movies_path,
                "a table that contains information about movies including titles, scores, ratings, genres, directors, and other metadata",
            )

            # Add reviews table
            db.add_text_table(
                "reviews",
                reviews_path,
                "a table that contains movie reviews with metadata and review text content for sentiment analysis",
            )

            # Link reviews to movies by movie id
            db.link("reviews", "movies", "id")

            # Build relevant values index for key categorical columns
            db.build_relevant_values_index(
                "movies", "genre", "director", "rating", "originalLanguage"
            )
            db.build_relevant_values_index(
                "reviews", "reviewState", "publicationName", "isTopCritic"
            )

        elif self.use_case == "detective":
            # For detective case, we might need to handle DuckDB files differently
            # For now, let's implement basic CSV loading
            dmv_path = str(self.data_path / "dmv_table.csv")
            evidence_path = str(self.data_path / "evidence_table.csv")
            shop_cams_path = str(self.data_path / "shop_cams_table.csv")
            traffic_cams_path = str(self.data_path / "traffic_cams_table.csv")

            db.add_tabular_table("dmv", dmv_path, "DMV records table")
            db.add_tabular_table("evidence", evidence_path, "Evidence table")
            db.add_tabular_table(
                "shop_cams", shop_cams_path, "Shop camera footage table"
            )
            db.add_tabular_table(
                "traffic_cams",
                traffic_cams_path,
                "Traffic camera footage table",
            )

        elif self.use_case == "animals":
            # Load animal data
            audio_path = str(self.data_path / "audio_data.csv")
            image_path = str(self.data_path / "image_data.csv")

            db.add_tabular_table(
                "audio_data", audio_path, "Animal audio data table"
            )
            db.add_tabular_table(
                "image_data", image_path, "Animal image data table"
            )

        else:
            raise ValueError(f"Unsupported use case: {self.use_case}")

        return db

    def get_system_name(self) -> str:
        """Return the name of the system."""
        return "caesura"

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query and return metric object with results.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)

        Returns:
            GenericQueryMetric object containing results DataFrame and metrics
        """
        # Create appropriate metric object
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        try:
            query_fn = self._discover_query_impl(query_id)
            start_time = time.time()
            results = query_fn()
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            metric.results = (
                results if isinstance(results, pd.DataFrame) else pd.DataFrame()
            )
            metric.status = "success"

            # Get token usage and cost from execution
            self._update_token_usage_and_cost(metric, results)

        except Exception as e:
            # Handle failure
            metric.status = "failed"
            metric.error = str(e)
            metric.results = self._get_empty_results_dataframe(query_id)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()

        return metric

    def _discover_query_impl(self, query_id: int):
        """
        Discover and return the implementation for a specific query.

        Args:
            query_id: ID of the query to find implementation for

        Returns:
            Callable query implementation method

        Raises:
            NotImplementedError: If query implementation is not found
        """
        method_name = f"_execute_q{query_id}"
        if hasattr(self, method_name):
            return getattr(self, method_name)
        else:
            raise NotImplementedError(
                f"Query Q{query_id} implementation not found. "
                f"Please implement {method_name} method."
            )

    def _discover_queries(self) -> List[int]:
        """
        Discover available queries for CAESURA.

        Any method named ``_execute_q<i>`` (where <i> is an integer ≥1) is treated
        as an implemented query.  The function returns the list of those integer
        IDs in ascending order.

        Returns:
            List of available query IDs
        """
        pattern = re.compile(r"_execute_q(\d+)$")
        query_ids: List[int] = []

        # `dir(self)` lists all attribute names visible on the instance;
        # we then pick out callables whose names match our pattern.
        for attr_name in dir(self):
            match = pattern.match(attr_name)
            if match:
                attr = getattr(self, attr_name, None)
                if callable(attr):
                    query_ids.append(int(match.group(1)))

        return sorted(query_ids)

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for a query.
        Should be overridden by scenario-specific runners.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with appropriate columns
        """
        # Generic empty DataFrame - should be overridden by subclasses
        return pd.DataFrame()

    def _update_token_usage_and_cost(
        self, metric: GenericQueryMetric, results: pd.DataFrame
    ):
        """Update metric with token usage and cost information."""
        try:
            # Get actual token usage from the CAESURA agent
            token_usage_dict = self.caesura_agent.get_token_usage()
            metric.token_usage = token_usage_dict.get("total_tokens", 0)
            metric.money_cost = self._calculate_actual_cost(token_usage_dict)

            # Print usage for debugging
            print(
                f"  Token usage: {metric.token_usage} tokens (prompt: {token_usage_dict.get('prompt_tokens', 0)}, "
                f"completion: {token_usage_dict.get('completion_tokens', 0)}), Cost: ${metric.money_cost:.4f}"
            )

        except Exception as e:
            print(f"  Warning: Could not get token usage: {e}")
            metric.token_usage = 0
            metric.money_cost = 0.0

    def _calculate_actual_cost(self, token_usage_dict: dict) -> float:
        """Calculate actual cost based on real token usage from API."""
        # Token pricing (USD per 1K tokens)
        pricing = {
            "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.002},
            "gpt-4-0613": {"input": 0.03, "output": 0.06},
        }

        if self.caesura_model in pricing:
            prompt_tokens = token_usage_dict.get("prompt_tokens", 0)
            completion_tokens = token_usage_dict.get("completion_tokens", 0)

            model_pricing = pricing[self.caesura_model]
            cost = (prompt_tokens / 1000) * model_pricing["input"] + (
                completion_tokens / 1000
            ) * model_pricing["output"]
            return round(cost, 6)

        return 0.0

    def execute_caesura_query(self, query_text: str) -> pd.DataFrame:
        """
        Execute a CAESURA natural language query and return results DataFrame.
        This method can be used by specific query implementations.

        Args:
            query_text: Natural language query string

        Returns:
            Results DataFrame
        """
        try:
            # Reset agent state for new query
            self.caesura_agent.reset_for_new_query()

            # Execute query using the shared agent
            self.caesura_agent.run(query_text)
            final_result = self.caesura_agent.get_final_result()

            # Extract results
            if final_result is not None and hasattr(final_result, "data_frame"):
                result_df = final_result.data_frame.copy()
                return result_df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"Error executing CAESURA query: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def get_available_use_cases(self) -> List[str]:
        """Get list of available use cases."""
        return ["movie", "detective", "animals"]

    def describe_dataset(self) -> str:
        """Get description of the current dataset."""
        return self.database.describe()
