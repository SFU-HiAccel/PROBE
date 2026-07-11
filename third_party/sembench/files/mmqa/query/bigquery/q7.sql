SELECT t.Airlines, i.uri
FROM mmqa.tampa_international_airport t, mmqa.images i
WHERE AI.IF(
    STRUCT(
        "You will be provided with an airline name and an image. ",
        "Determine if the image shows the logo of the airline. ",
        "Airline: ", t.Airlines, ", Image: ", i.uri
    ),
    connection_id => '<<connection>>', 
    model_params => JSON '{"labels": {"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>
);
