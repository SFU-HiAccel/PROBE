import os
import duckdb
import pandas as pd


class ThalamusDBEcommSetup:
    def setup_data(self, data_dir: str):
        db_path = os.path.join(data_dir, "thalamusdb.duckdb")
        if os.path.exists(db_path):
            os.remove(db_path)

        con = duckdb.connect(db_path)
        con.execute(f"set file_search_path = '{data_dir}'")
        con.execute(
            f"""CREATE OR REPLACE TABLE styles_details AS
            SELECT 
              *,
              -- ThalamusDB cannot execute semantic filters on expressions or multiple columns, so we have to manually concatenate and materialize them.
              -- Further, ThalamusDB cannot deal with columns containing strings with single quotes, so we remove them.
              replace(productDisplayName || ' ' || productDescriptors.description.value, '''', '') AS full_product_description
            FROM 'styles_details.parquet'
            """
        )
        con.execute(
            f"""CREATE OR REPLACE TABLE image_mapping AS
            SELECT
              *,
              '{os.path.join(data_dir, 'images', '')}' || filename AS local_image_path
            FROM 'image_mapping.parquet'
            """
        )
        con.close()
