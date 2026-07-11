SELECT p.patient_id
FROM medical_dataset.skin_cancer_mm as s, medical_dataset.patients AS p, medical_dataset.x_ray_mm  AS x 
WHERE p.patient_id = s.patient_id AND p.patient_id = x.patient_id AND AI.IF(
    prompt => (
    "You are given an image of a human skin mole and an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true only if both images indicate any diseases.", 
    s.image, x.image
    ), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>)
    