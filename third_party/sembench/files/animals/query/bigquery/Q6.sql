-- Q6: For which cities do we have images of monkeys but no audio recordings of monkeys?
-- Ground truth: select distinct city from ImageData I where Species LIKE '%MONKEY%' and not exists (select * from AudioData A where A.city = I.city and A.animal = 'Monkey');

SELECT DISTINCT I.City AS city
FROM animals_dataset.image_data_mm I 
WHERE AI.IF(('Does this image contain a monkey? ', I.image), 
    connection_id => '<<connection>>', 
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
AND NOT EXISTS (
    SELECT * FROM animals_dataset.audio_data_mm A 
    WHERE A.City = I.City 
    AND AI.IF(('Does this audio contain a monkey sound? ', A.audio), 
        connection_id => '<<connection>>', 
        model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
);

