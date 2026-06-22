-- Q5: What is the list of cities for which we have either images or audio recordings of elephants?
-- Ground truth: select distinct city from (select city from ImageData where Species LIKE '%ELEPHANT%') UNION (select city from AudioData where Animal = 'Elephant');

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
);
