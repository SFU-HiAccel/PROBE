WITH sick_audio AS (
  SELECT DISTINCT p.patient_id, age
  FROM medical_dataset.patients  AS p
  JOIN medical_dataset.audio_mm  AS a 
  ON p.patient_id = a.patient_id
  WHERE IF(AI.GENERATE(
    prompt => ("You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Classify if the patient is sick according to the audio (TRUE) or healthy (FALSE). Answer only TRUE or FALSE, nothing more.", 
    a.image), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>).result = "TRUE", 1, 0) = 1
),
sick_text AS (
  SELECT p.patient_id, age
  FROM medical_dataset.patients  AS p
  JOIN medical_dataset.symptoms_texts  AS s
  ON p.patient_id = s.patient_id
  WHERE IF(AI.GENERATE(
    FORMAT("""
    Classify if the patient is sick according to the symptoms (TRUE) or healthy (FALSE). 
    Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.
    Symptoms: %s.    
    Answer only TRUE or FALSE, nothing more.
    """, s.symptoms), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>).result = "TRUE", 1, 0) = 1
),
sick_image AS(
  SELECT p.patient_id, age
  FROM medical_dataset.x_ray_mm as x
  JOIN medical_dataset.patients  AS p
  ON p.patient_id = x.patient_id 
  WHERE IF(AI.GENERATE(
    prompt => ("You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Classify if the patient is sick according to the X-ray image (TRUE) or healthy (FALSE). Answer only TRUE or FALSE, nothing more.", 
    x.image), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>).result = "TRUE", 1, 0) = 1
),
sick_cancer AS(
  SELECT p.patient_id, age
  FROM medical_dataset.skin_cancer_mm as x
  JOIN medical_dataset.patients  AS p
  ON p.patient_id = x.patient_id
  WHERE IF(AI.GENERATE(
    prompt => ("You are given an image of human skin mole from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Classify if the mole or skin patch is mallicious/sick according to the image (TRUE) or benign/healthy (FALSE). Answer only TRUE or FALSE, nothing more.", 
    x.image), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>).result = "TRUE", 1, 0) = 1
)

SELECT sick_audio.patient_id FROM sick_audio
UNION DISTINCT
SELECT sick_text.patient_id FROM sick_text
UNION DISTINCT
SELECT sick_image.patient_id FROM sick_image
UNION DISTINCT
SELECT sick_cancer.patient_id FROM sick_cancer
