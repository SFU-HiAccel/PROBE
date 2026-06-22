SELECT p.patient_id, AI.GENERATE(
    ('Classify patient as sick or healthy on both the image of lung X-Ray:', x.image, ' and symptoms: ', s.symptoms, ' Answer only HEALTHY or SICK, nothing more.'), 
    connection_id => 'us.connection',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>) AS sick
FROM medical_dataset.patients  AS p, medical_dataset.symptoms_texts AS s, medical_dataset.x_ray_mm as x
WHERE p.patient_id = s.patient_id AND p.patient_id = x.patient_id