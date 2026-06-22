SELECT title
FROM mmqa.lizzy_caplan_text_data t
WHERE AI.IF(
  t.title || " is a comedy movie given their description: " || t.text,
  connection_id => '<<connection>>', 
  model_params => JSON '{"labels": {"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>
);
