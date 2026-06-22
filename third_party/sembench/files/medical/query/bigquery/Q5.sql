SELECT smoking_history, COUNT(*) AS count 
FROM
(
  SELECT DISTINCT p.patient_id, p.smoking_history
  FROM medical_dataset.patients AS p, medical_dataset.audio_mm AS a, medical_dataset.x_ray_mm AS x
  WHERE p.patient_id = x.patient_id AND p.patient_id = a.patient_id AND p.smoking_history = 'Current' AND
      AI.IF(
      prompt => (
        "You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if the recording captures an audio of sick lungs, with diseases.", 
        a.image
      ), 
      connection_id => '<<connection>>',
      model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
      <<other_params>>
      ) AND
      AI.IF(
      prompt => (
        "You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if there are lung problems (considered sick/disease) according to the X-ray image.", 
        x.image
      ), 
      connection_id => '<<connection>>',
      model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
      <<other_params>>
  )
)
GROUP BY smoking_history;
