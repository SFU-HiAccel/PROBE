import os

import pandas as pd
import lotus
from lotus.dtype_extensions import ImageArray

from src.runner.generic_lotus_runner.generic_lotus_runner import (
    GenericLotusRunner,
)

# Import additional modules for approximate policy
from lotus.models import SentenceTransformersRM
from lotus.types import CascadeArgs
from lotus.vector_store import FaissVS


class LotusRunner(GenericLotusRunner):
    """Runner for LOTUS system."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize LOTUS runner.

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

        # Initialize components for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            # Initialize both embedding models for mixed modality support
            self.rm_text = SentenceTransformersRM(model="intfloat/e5-base-v2")
            self.rm_image = SentenceTransformersRM("clip-ViT-B-32")
            self.vs = FaissVS()
            self.cascade_args = CascadeArgs(
                recall_target=0.8, precision_target=0.8
            )

    def _configure_lotus_for_join_type(self, join_type: str):
        """Configure LOTUS settings based on join type (text-only, image-only, or mixed)."""
        if hasattr(self, "policy") and self.policy == "approximate":
            if join_type == "text":
                lotus.settings.configure(
                    lm=self.lm, rm=self.rm_text, vs=self.vs
                )
            elif join_type == "image":
                lotus.settings.configure(
                    lm=self.lm, rm=self.rm_image, vs=self.vs
                )
            else:  # mixed or default
                # For mixed modality, use image embeddings as they handle both
                lotus.settings.configure(
                    lm=self.lm, rm=self.rm_image, vs=self.vs
                )

    def _execute_q1(self) -> pd.DataFrame:
        """
        Execute q1.

        Returns:
            DataFrame with columns: director
        """

        table_df = self.load_data("ben_piazza.csv", sep=",", quotechar='"')
        text_df = self.load_data(
            "ben_piazza_text_data.csv", sep=",", quotechar='"'
        )

        text_input_cols = ["text"]
        text_output_cols = {
            "director": "The director of the movie",
        }
        processed_text_df = text_df.sem_extract(
            text_input_cols,
            text_output_cols,
            extract_quotes=False,
            return_raw_outputs=False,
        )

        joined_df = pd.merge(
            table_df,
            processed_text_df,
            left_on="Title",
            right_on="title",
            how="left",
        )
        result_df = joined_df[joined_df["Role"] == "Bob Whitewood"]["director"]

        return result_df

    def _execute_q2a(self) -> pd.DataFrame:
        """
        Execute q2a.

        Returns:
            DataFrame with columns: ID, image_id
        """
        # Configure for image-only join (text in prompt is just column names, main comparison is visual)
        self._configure_lotus_for_join_type("image")

        table_df = self.load_data("ap_warrior.csv", sep=",", quotechar='"')
        image_dir = os.path.join(self.data_path, "images")
        image_filenames = os.listdir(image_dir)
        image_filepaths = []
        for image_id in image_filenames:
            if image_id.endswith(".png") or image_id.endswith(".jpg"):
                image_filepaths.append(os.path.join(image_dir, image_id))

        image_df = pd.DataFrame(
            {
                "image": ImageArray(image_filepaths),
                "image_filepath": image_filepaths,
            }
        )

        # Reset indices for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            image_df = image_df.reset_index(drop=True)
            table_df = table_df.reset_index(drop=True)

        prompt = "{image} shows the logo of horse racetrack {Track}"

        if hasattr(self, "policy") and self.policy == "approximate":
            result_df = image_df.sem_join(
                table_df,
                prompt,
                strategy="zs-cot",
                cascade_args=self.cascade_args
            )
        else:
            result_df = image_df.sem_join(table_df, prompt, strategy="zs-cot")

        result_df["image_id"] = result_df["image_filepath"].apply(
            lambda x: x.split("/")[-1]
        )

        return result_df[["ID", "image_id"]]

    def _execute_q2b(self) -> pd.DataFrame:
        """
        Execute q2b.

        Returns:
            DataFrame with columns: ID, image_id, color
        """
        # Configure for image-only join
        self._configure_lotus_for_join_type("image")

        table_df = self.load_data("ap_warrior.csv", sep=",", quotechar='"')
        image_dir = os.path.join(self.data_path, "images")
        image_filenames = os.listdir(image_dir)
        image_filepaths = []
        for image_id in image_filenames:
            if image_id.endswith(".png") or image_id.endswith(".jpg"):
                image_filepaths.append(os.path.join(image_dir, image_id))

        image_df = pd.DataFrame(
            {
                "image": ImageArray(image_filepaths),
                "image_filepath": image_filepaths,
            }
        )

        # Reset indices for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            image_df = image_df.reset_index(drop=True)
            table_df = table_df.reset_index(drop=True)

        prompt = "{image} shows the logo of horse racetrack {Track}"

        if hasattr(self, "policy") and self.policy == "approximate":
            result_df = image_df.sem_join(
                table_df,
                prompt,
                strategy="zs-cot",
                cascade_args=self.cascade_args
            )
        else:
            result_df = image_df.sem_join(table_df, prompt, strategy="zs-cot")

        result_df["image_id"] = result_df["image_filepath"].apply(
            lambda x: x.split("/")[-1]
        )

        input_cols = ["image"]
        output_cols = {
            "color": "The color of the logo in the image",
        }
        result_df = result_df.sem_extract(
            input_cols,
            output_cols,
            extract_quotes=False,
            return_raw_outputs=False,
        )

        return result_df[["ID", "image_id", "color"]]

    def _execute_q3a(self) -> pd.DataFrame:
        """
        Execute q3a.

        Returns:
            DataFrame with columns: title
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        prompt = "{title} is a comedy movie given their description: {text}"
        result_df = text_df.sem_filter(prompt)

        return result_df[["title"]]

    def _execute_q3b(self) -> pd.DataFrame:
        """
        Execute q3b.

        Returns:
            DataFrame with columns: title
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        prompt = "{title} is a sci-fi movie given their description: {text}"
        result_df = text_df.sem_filter(prompt)

        return result_df[["title"]]

    def _execute_q3c(self) -> pd.DataFrame:
        """
        Execute q3c.

        Returns:
            DataFrame with columns: title
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        prompt = "{title} is a romance movie given their description: {text}"
        result_df = text_df.sem_filter(prompt)

        return result_df[["title"]]

    def _execute_q3d(self) -> pd.DataFrame:
        """
        Execute q3d.

        Returns:
            DataFrame with columns: title
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        prompt = "{title} is a horror movie given their description: {text}"
        result_df = text_df.sem_filter(prompt)

        return result_df[["title"]]

    def _execute_q3e(self) -> pd.DataFrame:
        """
        Execute q3e.

        Returns:
            DataFrame with columns: title
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        prompt = "{title} is a heist movie given their description: {text}"
        result_df = text_df.sem_filter(prompt)

        return result_df[["title"]]

    def _execute_q3f(self) -> pd.DataFrame:
        """
        Execute q3f.

        Returns:
            DataFrame with columns: title
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        prompt = "{title} is a romantic comedy given their description: {text}"
        result_df = text_df.sem_filter(prompt)

        return result_df[["title"]]

    def _execute_q3g(self) -> pd.DataFrame:
        """
        Execute q3g.

        Returns:
            DataFrame with columns: title
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        prompt = (
            "{title} is a biographical comedy given their description: {text}"
        )
        result_df = text_df.sem_filter(prompt)

        return result_df[["title"]]

    def _execute_q4(self) -> pd.DataFrame:
        """
        Execute q4.

        Returns:
            DataFrame with columns: movies_in_genre
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        target_values = [
            "Orange County",
            "Mean Girls",
            "Love Is the Drug",
            "Crashing",
            "Cloverfield",
            "My Best Friend's Girl",
            "Crossing Over",
            "Hot Tub Time Machine",
            "The Last Rites of Ransom Pride",
            "127 Hours",
            "High Road",
            "Save the Date",
            "Bachelorette",
            "3, 2, 1... Frankie Go Boom",
            "Queens of Country",
            "Item 47",
            "The Interview",
            "The Night Before",
            "Now You See Me 2",
            "Allied",
            "The Disaster Artist",
            "Extinction",
            "The People We Hate at the Wedding",
            "Cobweb",
        ]
        text_df = text_df[text_df["title"].isin(target_values)]

        text_input_cols = ["text"]
        text_output_cols = {
            "genres": "The genres of the movie, separated by commas",
        }
        new_text_df = text_df.sem_extract(
            text_input_cols,
            text_output_cols,
            extract_quotes=False,
            return_raw_outputs=False,
        )
        print(new_text_df.head())

        expanded_data = []
        for _, row in new_text_df.iterrows():
            movie_title = row["title"]
            genres = [genre.strip() for genre in row["genres"].split(",")]
            for genre in genres:
                expanded_data.append({"genre": genre, "title": movie_title})

        df_expanded = pd.DataFrame(expanded_data)
        genre_movies_table = (
            df_expanded.groupby("genre")["title"]
            .apply(lambda x: ", ".join(x))
            .reset_index()
        )
        genre_movies_table.rename(
            columns={"title": "movies_in_genre"}, inplace=True
        )

        return genre_movies_table[["genre", "movies_in_genre"]]

    def _execute_q5(self) -> pd.DataFrame:
        """
        Execute q5.

        Returns:
            DataFrame with columns: _output
        """

        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        target_values = [
            "Love Is the Drug",
            "Crashing",
            "Cloverfield",
            "My Best Friend's Girl",
            "Hot Tub Time Machine",
            "The Last Rites of Ransom Pride",
            "Save the Date",
            "Bachelorette",
            "3, 2, 1... Frankie Go Boom",
            "Queens of Country",
            "Item 47",
            "The Night Before",
            "Now You See Me 2",
            "Allied",
            "Extinction",
            "Cobweb",
        ]
        text_df = text_df[text_df["title"].isin(target_values)]

        prompt = "Who has played a role in all the movies {title} listed in the table given their descriptions {text}? Simply give the name of the actor."  # noqa: E501
        result_df = text_df.sem_agg(prompt)

        return result_df[["_output"]]

    def _execute_q6a(self) -> pd.DataFrame:
        """
        Execute q6a.

        Returns:
            DataFrame with columns: Airlines
        """

        table_df = self.load_data(
            "tampa_international_airport.csv", sep=",", quotechar='"'
        )

        prompt = "Given destinations '{Destinations}' of {Airlines}, the airline has flights to Frankfurt"  # noqa: E501
        result_df = table_df.sem_filter(prompt)

        return result_df[["Airlines"]]

    def _execute_q6b(self) -> pd.DataFrame:
        """
        Execute q6b.

        Returns:
            DataFrame with columns: Airlines
        """

        table_df = self.load_data(
            "tampa_international_airport.csv", sep=",", quotechar='"'
        )

        prompt = "Given destinations '{Destinations}' of {Airlines}, the airline has flights to Germany"  # noqa: E501
        result_df = table_df.sem_filter(prompt)

        return result_df[["Airlines"]]

    def _execute_q6c(self) -> pd.DataFrame:
        """
        Execute q6c.

        Returns:
            DataFrame with columns: Airlines
        """

        table_df = self.load_data(
            "tampa_international_airport.csv", sep=",", quotechar='"'
        )

        prompt = "Given destinations '{Destinations}' of {Airlines}, the airline has flights to Europe"  # noqa: E501
        result_df = table_df.sem_filter(prompt)

        return result_df[["Airlines"]]

    def _execute_q7(self) -> pd.DataFrame:
        """
        Execute q7.

        Returns:
            DataFrame with columns: Airlines, image_id
        """
        # Configure for image-only join
        self._configure_lotus_for_join_type("image")

        table_df = self.load_data(
            "tampa_international_airport.csv", sep=",", quotechar='"'
        )

        image_dir = os.path.join(self.data_path, "images")
        image_filenames = os.listdir(image_dir)
        image_filepaths = []
        for image_id in image_filenames:
            if image_id.endswith(".png") or image_id.endswith(".jpg"):
                image_filepaths.append(os.path.join(image_dir, image_id))

        image_df = pd.DataFrame(
            {
                "image": ImageArray(image_filepaths),
                "image_filepath": image_filepaths,
            }
        )

        # Reset indices for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            table_df = table_df.reset_index(drop=True)
            image_df = image_df.reset_index(drop=True)

        prompt = "{image} shows the logo of {Airlines}"

        if hasattr(self, "policy") and self.policy == "approximate":
            result_df = table_df.sem_join(
                image_df,
                prompt,
                strategy="zs-cot",
                cascade_args=self.cascade_args
            )
        else:
            result_df = table_df.sem_join(image_df, prompt, strategy="zs-cot")

        result_df["image_id"] = result_df["image_filepath"].apply(
            lambda x: x.split("/")[-1]
        )

        return result_df[["Airlines", "image_id"]]
