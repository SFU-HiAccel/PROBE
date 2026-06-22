"""
DuckDB FlockMTL runner implementation.
"""

import time
from typing import Dict, List

from jinja2 import Environment
from overrides import override

from runner.generic_runner import GenericQueryMetric, GenericRunner

jinja_env = Environment(variable_start_string="<<", variable_end_string=">>")


class GenericFlockMTLRunner(GenericRunner):
    """Runner for FlockMTL."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize DuckDB FlockMTL runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        self.flockmtl_conn = None

    @override
    def get_system_name(self) -> str:
        return "flockmtl"

    @override
    def execute_queries(
        self, query_ids: List[int]
    ) -> Dict[int, GenericQueryMetric]:
        query_texts = {
            query_id: (
                self._discover_query_text(query_id)
                if self.scenario_handler is None
                else self.scenario_handler.get_query_text(
                    query_id, self.get_system_name()
                )
            )
            for query_id in query_ids
        }
        query_metrics = {
            query_id: GenericQueryMetric(query_id=query_id, status="pending")
            for query_id in query_ids
        }

        for query_id, query_text in query_texts.items():
            try:
                # Replace variable names in the query text
                templated_query = jinja_env.from_string(query_text).render(
                    model_name=self.model_name
                )

                print(templated_query)

                start_time = time.time()
                query_job = self.flockmtl_conn.execute(templated_query)
                df = query_job.fetchdf()
                execution_time = time.time() - start_time

                query_metrics[query_id].results = df
                query_metrics[query_id].execution_time = execution_time
                query_metrics[query_id].status = "success"
            except Exception as e:
                print(
                    f"  Error executing query {query_id}: {type(e).__name__}: {e}"  # noqa: E501
                )
                query_metrics[query_id].status = "failed"
                query_metrics[query_id].error = str(e)
        print(query_metrics.items())
        for query_id, metrics in query_metrics.items():
            if metrics.status != "failed":
                # TODO: Implement token usage and cost calculation
                # Currently no way to extract token usage from FlockMTL
                metrics.token_usage = 0
                metrics.money_cost = 0.0

        return query_metrics
