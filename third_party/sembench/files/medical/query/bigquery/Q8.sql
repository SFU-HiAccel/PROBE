SELECT p.patient_id
FROM medical_dataset.skin_cancer_mm as x
JOIN medical_dataset.patients  AS p
ON p.patient_id = x.patient_id 
WHERE p.did_family_have_cancer = 1 AND AI.IF(
    prompt => (
    "You are given an image of human skin mole from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if mole or skin area is malignant (considered abnormal/cancerous/sick) according to the image.", 
    x.image
    ), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>)
LIMIT 100