-- Q7: What are the cities for which zebras and impala co-occur?
-- Ground truth: select city from (select city from ImageData where Species LIKE '%ZEBRA%') INTERSECT (select city from ImageData where Species LIKE '%IMPALA%');
SELECT City AS city FROM (
    SELECT DISTINCT City
    FROM animals_dataset.image_data_mm 
    WHERE AI.IF(('Does this image contain a zebra?', image), 
        connection_id => '<<connection>>', 
        model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
) INTERSECT DISTINCT (
    SELECT DISTINCT City
    FROM animals_dataset.image_data_mm 
    WHERE AI.IF(('Does this image contain an impala?', image), 
        connection_id => '<<connection>>', 
        model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
);