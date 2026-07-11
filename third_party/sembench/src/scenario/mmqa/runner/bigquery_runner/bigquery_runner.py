from pathlib import Path

from runner.generic_bigquery_runner.generic_bigquery_runner import (
    GenericBigQueryRunner,
)
from scenario.mmqa.setup.bigquery import BigQueryMMQASetup


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

        if not skip_setup:
            self._bigquery_setup()

    def _bigquery_setup(self):
        setup = BigQueryMMQASetup()
        setup.setup_data(
            data_dir=Path(__file__).resolve().parents[5]
            / "files"
            / "mmqa"
            / "data"
            / f"sf_{self.scale_factor}"
        )
