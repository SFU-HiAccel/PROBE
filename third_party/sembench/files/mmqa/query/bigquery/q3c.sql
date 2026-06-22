SELECT title
FROM mmqa.lizzy_caplan_text_data t
WHERE AI.IF(
  t.title || " is a romance movie given their description: " || t.text,
  connection_id => '<<connection>>', 
  endpoint => '<<endpoint>>'
);
