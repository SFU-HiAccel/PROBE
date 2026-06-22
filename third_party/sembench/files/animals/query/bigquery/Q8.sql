-- Q8: What is the list of cities for which we found at least one image or audio recording of elephants and at least one image or audio recording of monkeys?
-- Ground truth: select city from ((select city from ImageData where Species LIKE '%ELEPHANT%') UNION (select city from AudioData where Animal = 'Elephant'))  INTERSECT ((select city from ImageData where Species LIKE '%MONKEY%') UNION (select city from AudioData where Animal = 'Monkey'));

SELECT City AS city FROM (
    SELECT DISTINCT city FROM (
        SELECT City AS city
        FROM animals_dataset.image_data_mm 
        WHERE AI.IF(('Does this image contain an elephant? ', image), 
            connection_id => '<<connection>>', 
            model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
        UNION ALL
        SELECT City AS city
        FROM animals_dataset.audio_data_mm 
        WHERE AI.IF(('Does this audio contain an elephant sound? ', audio), 
            connection_id => '<<connection>>', 
            model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
    )
) INTERSECT DISTINCT (
    SELECT DISTINCT city FROM (
        SELECT City AS city
        FROM animals_dataset.image_data_mm 
        WHERE AI.IF(('Does this image contain a monkey? ', image), 
            connection_id => '<<connection>>', 
            model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
        UNION ALL
        SELECT City AS city
        FROM animals_dataset.audio_data_mm 
        WHERE AI.IF(('Does this audio contain a monkey sound? ', audio), 
            connection_id => '<<connection>>', 
            model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
    )
);

