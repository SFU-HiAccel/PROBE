SELECT p.patient_id
FROM medical_dataset.patients  AS p
JOIN medical_dataset.symptoms_texts  AS s
ON p.patient_id = s.patient_id
WHERE AI.IF(
    FORMAT("""
    Patient with these symptoms has an allergy. Symptoms: %s. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.
    """, s.symptoms), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>
)
