WITH movie_with_director AS (
    SELECT t.title, AI.GENERATE(
        prompt => ("Extract the director name from the following movie description: ", t.text),
        connection_id => '<<connection>>',
        model_params => JSON '{"labels": {"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>
    ).result AS director
    FROM mmqa.ben_piazza_text_data t
)

SELECT t2.director
FROM mmqa.ben_piazza t1
JOIN movie_with_director t2
ON t1.Title = t2.title
WHERE t1.Role = "Bob Whitewood";
