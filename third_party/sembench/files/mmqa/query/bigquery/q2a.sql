SELECT t.ID, i.uri
FROM mmqa.ap_warrior t, mmqa.images i
WHERE AI.IF(
  STRUCT(
    "You will be provided with a horse racetrack name and an image. ",
    "Determine if the image shows the logo of the racetrack. ",
    "Racetrack: ", t.Track, ", Image: ", i.uri
  ),
  connection_id => '<<connection>>',
  model_params => JSON '{"labels": {"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>
);
