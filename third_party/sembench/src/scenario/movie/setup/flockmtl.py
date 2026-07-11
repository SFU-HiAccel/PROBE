

import os
from pathlib import Path

import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

MOVIE_FILES_DIR = os.path.abspath(
    Path(__file__).resolve().parents[4] / "files" / "movie" / "data"
)

class FlockMTLMovieSetup:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the FlockMTL connection using environment variables.
        """
        if os.environ.get('OPENAI_API_KEY') is None:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")

        self.flockmtl_conn = duckdb.connect(os.path.join(MOVIE_FILES_DIR, "movie_database.duckdb"))
        
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
        

    def setup_data(self, data_dir: str):
        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/Movies_2000.csv"), 
            table_name="movies_2000"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/Reviews_2000.csv"), 
            table_name="reviews_2000"
        )


    def get_connection(self):
        """
        Returns the FlockMTL connection.
        """
        return self.flockmtl_conn
