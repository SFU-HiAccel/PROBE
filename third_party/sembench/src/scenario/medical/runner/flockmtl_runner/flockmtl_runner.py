"""
FlockMTL runner implementation.
"""

from pathlib import Path
import sys

from typing import Any, Dict, List
from overrides import override
import pandas as pd
from pathlib import Path
import os
import time
from jinja2 import Environment

from runner.generic_runner import GenericQueryMetric
from scenario.medical.setup.flockmtl import FlockMTLMedicalSetup
from runner.generic_flockmtl_runner.generic_flockmtl_runner import (
    GenericFlockMTLRunner,
)

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

jinja_env = Environment(variable_start_string="<<", variable_end_string=">>")


class FlockMTLRunner(GenericFlockMTLRunner):
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
            skip_setup=skip_setup,
        )

        self.flockmtl_conn = FlockMTLMedicalSetup().get_connection()
