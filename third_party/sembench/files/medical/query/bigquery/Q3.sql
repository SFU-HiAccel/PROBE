SELECT age, gender, smoking_history, did_family_have_cancer, p.patient_id 
FROM medical_dataset.patients as p 
JOIN medical_dataset.x_ray_mm  AS x 
ON p.patient_id = x.patient_id
WHERE p.did_family_have_cancer = 1 AND AI.IF(
    prompt => (
      "You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if there are lung problems (considered sick/disease) according to the X-ray image.", 
      x.image
    ), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>
)
LIMIT 5
