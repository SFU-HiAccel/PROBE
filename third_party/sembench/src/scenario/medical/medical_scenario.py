import os
from typing import List

from scenario.medical.preparation.generate_data import prepare_data
import glob


MEDICAL_FILES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "files", "medical"
    )
)


class MedicalScenario:
    """
    Medical scenario handler.

    This class:
     * downloads and prepares data
     * retrievs queries
    """

    def __init__(self, scale_factor: int = 11112):
        self.data_dir = MEDICAL_FILES_DIR
        self.scale_factor = scale_factor

    def setup_scenario(self, systems: List[str]) -> None:
        # Download and prepare data if not already done
        prepare_data(scaling_factor=self.scale_factor)

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery":
                from scenario.medical.setup.bigquery import BigQueryMedicalSetup

                setup = BigQueryMedicalSetup()
                setup.setup_data(data_dir=self.data_dir, scale_factor=self.scale_factor)

            elif system == "snowflake":
                from scenario.medical.setup.snowflake import (
                    SnowflakeMedicalSetup,
                )

                setup = SnowflakeMedicalSetup()
                setup.setup_data(self.data_dir, self.scale_factor)
            
            elif system == "flockmtl":
                from scenario.medical.setup.flockmtl import FlockMTLMedicalSetup

                setup = FlockMTLMedicalSetup()
                setup.setup_data(self.data_dir, self.scale_factor)
                
            elif system == "lotus":
                pass  # Nothing to do. LOTUS works on raw files.

            elif system == "palimpzest":
                pass  # Nothing to do. Palimpzest works on raw files.
            
            elif system == "thalamusdb":
                pass

            else:
                raise ValueError(f"Unsupported system: {system}")

    def get_query_text(self, query_id: int, system_name: str) -> str:
        """
        Get the SQL query text for a given query ID and system name.

        Args:
            query_id: ID of the query
            system_name: Name of the system
        Returns:
            SQL query text as a string
        """
        system_query_dir = os.path.abspath(
            os.path.join(MEDICAL_FILES_DIR, "query", system_name)
        )
        matching_files = glob.glob(
            os.path.join(system_query_dir, f"Q{query_id}.*")
        )

        if not matching_files:
            print(os.path.join(system_query_dir, f"Q{query_id}.*"))
            raise FileNotFoundError(
                f"No query implementation found for query {query_id} and system '{system_name}'"
            )
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple query files found for query {query_id} and system '{system_name}': {matching_files}"
            )

        with open(matching_files[0], "r") as f:
            return f.read()

    def get_data_dir(self) -> str:
        return self.data_dir
