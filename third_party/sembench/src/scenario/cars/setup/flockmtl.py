

import os
from pathlib import Path

import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

CARS_FILES_DIR = os.path.abspath(
    Path(__file__).resolve().parents[4] / "files" / "cars" / "data"
)


class FlockMTLCarsSetup:
    def __init__(self, model_name: str = "gpt-4o-mini", db_name:str = 'cars_database', load_extensions: bool = True, db_folder: str = CARS_FILES_DIR):
        """
        Initializes the FlockMTL connection using environment variables.
        """
        if os.environ.get('OPENAI_API_KEY') is None:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")

        self.flockmtl_conn = duckdb.connect(os.path.join(db_folder, f"{db_name}.duckdb"))

        if load_extensions:
            self.flockmtl_conn.install_extension("flockmtl", repository="community")
            self.flockmtl_conn.load_extension("flockmtl")

            self.flockmtl_conn.execute(
                f"""CREATE SECRET (TYPE OPENAI,API_KEY '{os.environ.get('OPENAI_API_KEY')}');"""
            )

            if not model_name in self.flockmtl_conn.execute("GET MODELS;").fetchdf()["model"].tolist():
                self.flockmtl_conn.execute("""
                    CREATE MODEL(
                    'model_name',
                    'model_name',
                    'openai',
                    {"tuple_format": "json", "batch_size": 32, "model_parameters": {"temperature": 0.7}}
                    );
                """.replace("model_name", model_name))


    def _upload_file_to_db(self, csv_path: str, table_name: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}. Please run the download script first.")

        self.flockmtl_conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}');
            """)


    def setup_data(self, data_dir: str, scale_factor: int = 157376):
        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data", f"car_data_{scale_factor}.csv"),
            table_name="cars"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data", f"audio_car_data_{scale_factor}.csv"),
            table_name="car_audio"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data", f"text_complaints_data_{scale_factor}.csv"),
            table_name="car_complaints"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data", f"image_car_data_{scale_factor}.csv"),
            table_name="car_images"
        )

    def get_connection(self):
        """
        Returns the FlockMTL connection.
        """
        return self.flockmtl_conn

    def run_query(self, query: str):
        self.flockmtl_conn.execute(query)
