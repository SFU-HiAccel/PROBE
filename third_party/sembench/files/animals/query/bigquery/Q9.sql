-- Q9: What is the list of cities for which we have both images and audio recordings of monkeys?
-- Ground truth: select distinct city from (select city from ImageData where Species LIKE '%MONKEY%') INTERSECT (select city from AudioData where Animal = 'Monkey');

SELECT DISTINCT City AS city FROM (
    SELECT City
    FROM animals_dataset.image_data_mm 
    WHERE AI.IF(('Does this image contain a monkey? ', image), 
        connection_id => '<<connection>>', 
        model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
) INTERSECT DISTINCT (
    SELECT City
    FROM animals_dataset.audio_data_mm 
    WHERE AI.IF(('Does this audio contain a monkey sound? ', audio), 
        connection_id => '<<connection>>', 
        model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
);

