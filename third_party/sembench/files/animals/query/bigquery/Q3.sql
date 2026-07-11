-- Q3: What is the city where we captured most pictures of zebras?
-- Ground truth: select city from ImageData where Species LIKE '%ZEBRA%' group by city order by count(*) desc limit 1;
SELECT City AS city
FROM animals_dataset.image_data_mm 
WHERE AI.IF(('Does this image contain a zebra?', image), 
    connection_id => '<<connection>>', 
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
GROUP BY City 
ORDER BY COUNT(*) DESC 
LIMIT 1;