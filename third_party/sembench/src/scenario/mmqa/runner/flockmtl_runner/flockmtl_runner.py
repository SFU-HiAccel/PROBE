"""
FlockMTL system runner implementation for MMQA scenario.
"""

from src.scenario.mmqa.setup.flockmtl import FlockMTLMMQASetup
from src.runner.generic_flockmtl_runner.generic_flockmtl_runner import (
    GenericFlockMTLRunner,
)


class FlockMTLRunner(GenericFlockMTLRunner):
    """Runner for DuckDB FlockMTL."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize FlockMTL runner.

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

        setup = FlockMTLMMQASetup(model_name=model_name)
        if not skip_setup:
            setup.setup_data(data_dir=self.data_path)
        self.flockmtl_conn = setup.get_connection()

    # def _execute_q1(self) -> pd.DataFrame:
    #     """
    #     Execute q1.

    #     Returns:
    #         DataFrame with columns: director
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('extract_director', 'Extract the director name from the following movie description: {{description}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(self.data_path, "ben_piazza_text_data.csv")
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT
    #             title,
    #             text,
    #             llm_complete(
    #                 {'model_name': {self.model_name}},
    #                 {'prompt_name': 'extract_director'},
    #                 {'description': text}
    #             ) AS director
    #         FROM movies;
    #         """
    #     ).fetchdf()

    #     table_df = self.load_data("ben_piazza.csv", sep=",", quotechar='"')
    #     joined_df = pd.merge(
    #         table_df, result_df, left_on="Title", right_on="title", how="left"
    #     )

    #     return joined_df[joined_df["Role"] == "Bob Whitewood"]["director"]

    # def _execute_q2a(self):
    #     raise NotImplementedError(
    #         "FlockMTL hasn't supported semantic joins between texts and images."
    #     )

    # def _execute_q2b(self):
    #     raise NotImplementedError(
    #         "FlockMTL hasn't supported semantic joins between texts and images."
    #     )

    # def _execute_q3a(self) -> pd.DataFrame:
    #     """
    #     Execute q3a.

    #     Returns:
    #         DataFrame with columns: title
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_genre', '{{title}} is a comedy movie given their description: {{text}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT title
    #         FROM movies
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_genre'},
    #             {'title': title, 'text': text}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_genre;")

    #     return result_df[["title"]]

    # def _execute_q3b(self) -> pd.DataFrame:
    #     """
    #     Execute q3b.

    #     Returns:
    #         DataFrame with columns: title
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_genre', '{{title}} is a sci-fi movie given their description: {{text}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT title
    #         FROM movies
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_genre'},
    #             {'title': title, 'text': text}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_genre;")

    #     return result_df[["title"]]

    # def _execute_q3c(self) -> pd.DataFrame:
    #     """
    #     Execute q3c.

    #     Returns:
    #         DataFrame with columns: title
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_genre', '{{title}} is a romance movie given their description: {{text}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT title
    #         FROM movies
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_genre'},
    #             {'title': title, 'text': text}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_genre;")

    #     return result_df[["title"]]

    # def _execute_q3d(self) -> pd.DataFrame:
    #     """
    #     Execute q3d.

    #     Returns:
    #         DataFrame with columns: title
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_genre', '{{title}} is a horror movie given their description: {{text}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT title
    #         FROM movies
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_genre'},
    #             {'title': title, 'text': text}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_genre;")

    #     return result_df[["title"]]

    # def _execute_q3e(self) -> pd.DataFrame:
    #     """
    #     Execute q3e.

    #     Returns:
    #         DataFrame with columns: title
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_genre', '{{title}} is a heist movie given their description: {{text}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT title
    #         FROM movies
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_genre'},
    #             {'title': title, 'text': text}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_genre;")

    #     return result_df[["title"]]

    # def _execute_q3f(self) -> pd.DataFrame:
    #     """
    #     Execute q3f.

    #     Returns:
    #         DataFrame with columns: title
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_genre', '{{title}} is a romantic comedy given their description: {{text}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT title
    #         FROM movies
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_genre'},
    #             {'title': title, 'text': text}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_genre;")

    #     return result_df[["title"]]

    # def _execute_q3g(self) -> pd.DataFrame:
    #     """
    #     Execute q3g.

    #     Returns:
    #         DataFrame with columns: title
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_genre', '{{title}} is a biographical comedy given their description: {{text}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT title
    #         FROM movies
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_genre'},
    #             {'title': title, 'text': text}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_genre;")

    #     return result_df[["title"]]

    # def _execute_q4(self) -> pd.DataFrame:
    #     """
    #     Execute q4.

    #     Returns:
    #         DataFrame with columns: movies_in_genre
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('extract_genres', 'Extract genres of each movie, separated by commas, given their description: {{text}}');
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}');"  # noqa: E501
    #     )

    #     results_df = self.con.execute(
    #         f"""
    #         SELECT
    #             title,
    #             llm_complete(
    #                 {'model_name': {self.model_name}},
    #                 {'prompt_name': 'extract_genres'},
    #                 {'text': text}
    #             ) AS genres
    #         FROM movies;
    #         """
    #     ).fetchdf()

    #     expanded_data = []
    #     for _, row in results_df.iterrows():
    #         movie_title = row["title"]
    #         genres = [genre.strip() for genre in row["genres"].split(",")]
    #         for genre in genres:
    #             expanded_data.append({"genre": genre, "title": movie_title})

    #     expanded_df = pd.DataFrame(expanded_data)
    #     genre_movies_table = (
    #         expanded_df.groupby("genre")["title"]
    #         .apply(lambda x: ", ".join(x))
    #         .reset_index()
    #     )
    #     genre_movies_table.rename(
    #         columns={"title": "movies_in_genre"}, inplace=True
    #     )

    #     return genre_movies_table[["movies_in_genre"]]

    # def _execute_q5(self) -> pd.DataFrame:
    #     """
    #     Execute q5.

    #     Returns:
    #         DataFrame with columns: actor
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('summarize_actor', 'Who has played a role in all the movies {{title}} listed in the table given their descriptions {{text}}? Simply give the name of the actor.);
    #         """  # noqa: E501
    #     )
    #     text_filepath = os.path.join(
    #         self.data_path, "lizzy_caplan_text_data.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE movies AS SELECT * FROM read_csv_auto('{text_filepath}') WHERE title IN ('Love Is the Drug', 'Crashing', 'Cloverfield', 'My Best Friend's Girl', 'Hot Tub Time Machine', 'The Last Rites of Ransom Pride', 'Save the Date', 'Bachelorette', '3, 2, 1... Frankie Go Boom', 'Queens of Country', 'Item 47', 'The Night Before', 'Now You See Me 2', 'Allied', 'Extinction', 'Cobweb');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT
    #             llm_reduce(
    #                 {'model_name': {self.model_name}},
    #                 {'prompt_name': 'summarize_actor'},
    #                 {'title': title, 'text': text}
    #             ) AS actor
    #         FROM movies;
    #         """
    #     ).fetchdf()

    #     return result_df[["actor"]]

    # def _execute_q6a(self) -> pd.DataFrame:
    #     """
    #     Execute q6a.

    #     Returns:
    #         DataFrame with columns: Airlines
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_destination', 'Given destinations {{Destinations}} of {{Airlines}}, does {{Airlines}} have flights to Frankfurt?');
    #         """  # noqa: E501
    #     )
    #     table_filepath = os.path.join(
    #         self.data_path, "tampa_international_airport.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE tampa_airport AS SELECT * FROM read_csv_auto('{table_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT Airlines
    #         FROM tampa_airport
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_destination'},
    #             {'Airlines': Airlines, 'Destinations': Destinations}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_destination;")

    #     return result_df[["Airlines"]]

    # def _execute_q6b(self) -> pd.DataFrame:
    #     """
    #     Execute q6b.

    #     Returns:
    #         DataFrame with columns: Airlines
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_destination', 'Given destinations {{Destinations}} of {{Airlines}}, does {{Airlines}} have flights to Germany?');
    #         """  # noqa: E501
    #     )
    #     table_filepath = os.path.join(
    #         self.data_path, "tampa_international_airport.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE tampa_airport AS SELECT * FROM read_csv_auto('{table_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT Airlines
    #         FROM tampa_airport
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_destination'},
    #             {'Airlines': Airlines, 'Destinations': Destinations}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_destination;")

    #     return result_df[["Airlines"]]

    # def _execute_q6c(self) -> pd.DataFrame:
    #     """
    #     Execute q6c.

    #     Returns:
    #         DataFrame with columns: Airlines
    #     """

    #     self.con.execute(
    #         """
    #         CREATE PROMPT('filter_by_destination', 'Given destinations {{Destinations}} of {{Airlines}}, does {{Airlines}} have flights to Europe?');
    #         """  # noqa: E501
    #     )
    #     table_filepath = os.path.join(
    #         self.data_path, "tampa_international_airport.csv"
    #     )
    #     self.con.execute(
    #         f"CREATE TABLE tampa_airport AS SELECT * FROM read_csv_auto('{table_filepath}');"  # noqa: E501
    #     )

    #     result_df = self.con.execute(
    #         f"""
    #         SELECT Airlines
    #         FROM tampa_airport
    #         WHERE llm_filter(
    #             {'model_name': {self.model_name}},
    #             {'prompt_name': 'filter_by_destination'},
    #             {'Airlines': Airlines, 'Destinations': Destinations}
    #         );
    #         """
    #     ).fetchdf()
    #     self.con.execute("DELETE PROMPT filter_by_destination;")

    #     return result_df[["Airlines"]]

    # def _execute_q7(self):
    #     raise NotImplementedError(
    #         "FlockMTL hasn't supported semantic joins between texts and images."
    #     )
