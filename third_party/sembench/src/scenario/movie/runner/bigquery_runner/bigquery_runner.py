"""
Created on August 4, 2025

@author: Jiale Lao

Movie BigQuery system runner implementation.
"""

import time
from typing import Dict, List, override
import pandas as pd
from pathlib import Path
import os
from google.cloud import bigquery

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_runner import GenericQueryMetric
from runner.generic_bigquery_runner.generic_bigquery_runner import (
    GenericBigQueryRunner,
)

from scenario.movie.setup.bigquery import BigQueryMovieSetup


class BigQueryRunner(GenericBigQueryRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = True,
    ):
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        # Note: BigQuery setup is handled by scenario_handler.setup_scenario()
        # in GenericRunner.__init__() when skip_setup=False
