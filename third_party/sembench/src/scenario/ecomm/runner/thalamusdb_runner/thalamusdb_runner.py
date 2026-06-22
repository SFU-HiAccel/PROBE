"""
ThalamusDB system runner implementation.
"""

import os
from pathlib import Path
import sys
from typing import override

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_thalamusdb_runner.generic_thalamusdb_runner import (
    GenericThalamusDBRunner,
)


class ThalamusDBRunner(GenericThalamusDBRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            None,
            skip_setup=skip_setup,
        )

    @override
    def _discover_query_impl(self, query_id) -> callable:
        sql_text = self.scenario_handler.get_query_text(
            query_id, self.get_system_name()
        )
        return lambda _: self.execute_thalamusdb_query(sql_text)
