SELECT DISTINCT p.patient_id 
FROM medical_dataset.patients  AS p
JOIN medical_dataset.audio_mm  AS a 
ON p.patient_id = a.patient_id
WHERE p.smoking_history != 'Current' AND AI.IF(
    prompt => (
      "You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if the recording is healthy lungs, without diseases.", 
      a.image
    ), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>
)
