import argparse
import os

import lotus
import pandas as pd
from lotus.models import LM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="vertex_ai/meta/llama-3.3-70b-instruct-maas",
        help="Identifier for the language model to use",
    )

    parser.add_argument(
        "--table_filepath",
        type=str,
        default="/Users/tianjicong/Desktop/MMBench-System/files/mmqa/data/ben_piazza.csv",  # noqa: E501
        help="Path to the CSV file containing table data",
    )

    parser.add_argument(
        "--text_filepath",
        type=str,
        default="/Users/tianjicong/Desktop/MMBench-System/files/mmqa/data/ben_piazza_text_data.csv",  # noqa: E501
        help="Path to the CSV file containing text data",
    )

    parser.add_argument(
        "--output_filepath",
        type=str,
        default="/Users/tianjicong/Desktop/MMQA/execution_results/lotus/q1_llama3_3.csv",  # noqa: E501
        help="Path to the output file",
    )

    args = parser.parse_args()

    os.environ["VERTEXAI_PROJECT"] = "bq-mm-benchmark"
    os.environ["VERTEXAI_LOCATION"] = "us-central1"

    lm = LM(model=args.model_name)
    lotus.settings.configure(lm=lm)

    table_df = pd.read_csv(args.table_filepath, sep=",", quotechar='"')
    text_df = pd.read_csv(args.text_filepath, sep=",", quotechar='"')

    prompt = "Analyze the movie description and extract the director name."
    text_input_cols = ["text"]
    text_output_cols = {
        "director": "The director of the movie",
    }

    processed_text_df = text_df.sem_extract(
        text_input_cols,
        text_output_cols,
        # extract_quotes=False,
        # return_raw_outputs=False,
    )
    print(processed_text_df.head())
    joined_df = pd.merge(
        table_df,
        processed_text_df,
        left_on="Title",
        right_on="title",
        how="left",
    )
    result_df = joined_df[joined_df["Role"] == "Bob Whitewood"]["director"]
    result_df.to_csv(args.output_filepath, index=False)
