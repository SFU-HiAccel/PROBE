WITH movie_genres AS (
    SELECT t1.title, AI.GENERATE(
        prompt => ("Extract all applicable genres for each movie based on their description: ", t1.text),
        connection_id => '<<connection>>',
        output_schema => "genres ARRAY<STRING>",
        model_params => JSON '{"labels": {"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>
    ).genres
    FROM mmqa.lizzy_caplan_text_data AS t1
    WHERE t1.title IN (
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
        "Cobweb"
    )
)

SELECT
  unnested_genre AS genre,
  STRING_AGG(t2.title, ', ') AS movies_in_genre
FROM
  movie_genres AS t2,
  UNNEST(t2.genres) AS unnested_genre
GROUP BY
  unnested_genre
ORDER BY
  unnested_genre;
