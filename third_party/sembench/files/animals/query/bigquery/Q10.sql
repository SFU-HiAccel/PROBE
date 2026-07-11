-- Q10: What is the city and station with most associated pictures showing zebras?
-- Ground truth: select city, stationID from ImageData where Species LIKE '%ZEBRA%'  group by city, stationID order by count(*) DESC limit 1;
SELECT City AS city, StationID AS stationID
FROM animals_dataset.image_data_mm 
WHERE AI.IF(('Does this image contain a zebra?', image), 
    connection_id => '<<connection>>', 
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>)
GROUP BY City, StationID 
ORDER BY COUNT(*) DESC 
LIMIT 1;