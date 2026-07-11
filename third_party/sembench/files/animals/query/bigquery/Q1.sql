-- Q1: How many pictures of zebras do we have in our database?
-- Ground truth: select count(*) from ImageData where Species LIKE '%ZEBRA%';
SELECT COUNT(*) AS count
FROM animals_dataset.image_data_mm 
WHERE AI.IF(('Does this image contain a zebra?', image), 
    connection_id => '<<connection>>', 
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>);