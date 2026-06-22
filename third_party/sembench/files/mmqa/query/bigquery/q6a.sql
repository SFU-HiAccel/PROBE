SELECT Airlines
FROM mmqa.tampa_international_airport
WHERE AI.IF(
    "Given destinations '" || Destinations || "' of " || Airlines || ", the airline has flights to Frankfurt.",
    connection_id => '<<connection>>', 
    model_params => JSON '{"labels": {"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>
);
