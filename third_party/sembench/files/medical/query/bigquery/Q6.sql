WITH audio_denorm AS (
  SELECT * FROM (SELECT
  patient_id,
  location,
  MAX(IF(filtration_type = 'bell', image.uri, NULL)) AS bell_audio,
  MAX(IF(filtration_type = 'bell', audio_id, NULL)) AS bell_audio_id,
  MAX(IF(filtration_type = 'extended', image.uri, NULL)) AS extended_audio,
  MAX(IF(filtration_type = 'extended', audio_id, NULL)) AS extended_audio_id,
  MAX(IF(filtration_type = 'diaphragm', image.uri, NULL)) AS diaphragm_audio,
  MAX(IF(filtration_type = 'diaphragm', audio_id, NULL)) AS diaphragm_audio_id
  FROM medical_dataset.audio_mm
  GROUP BY patient_id, location)),
two_more_modalities AS (
  SELECT p.patient_id, p.age, s.symptom_id, s.symptoms, x.xray_id, x.image as image, a.bell_audio_id, a.bell_audio, a.extended_audio_id, a.extended_audio, a.diaphragm_audio_id, a.diaphragm_audio
  FROM medical_dataset.patients as p 
  LEFT JOIN audio_denorm AS a ON p.patient_id = a.patient_id
  LEFT JOIN medical_dataset.symptoms_texts as s ON p.patient_id = s.patient_id
  LEFT JOIN medical_dataset.x_ray_mm as x ON p.patient_id = x.patient_id
  WHERE (a.bell_audio_id IS NOT NULL AND s.symptom_id IS NOT NULL) OR
    (x.xray_id IS NOT NULL AND s.symptom_id IS NOT NULL) OR
    (x.xray_id IS NOT NULL AND a.bell_audio_id IS NOT NULL) 
),
sick_audio AS (
  SELECT a.patient_id
  FROM two_more_modalities as a
  WHERE a.bell_audio_id IS NOT NULL AND AI.IF(
        prompt => (
          "You are given three audio recordings of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if the recording captures sick lungs, with diseases.", 
          a.bell_audio, a.extended_audio, a.diaphragm_audio
        ), 
        connection_id => '<<connection>>',
        model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>)),
sick_image AS(
  SELECT x.patient_id
  FROM two_more_modalities as x
  WHERE x.xray_id IS NOT NULL AND AI.IF(
      prompt => (
        "You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if there are lung problems (considered sick/disease) according to the X-ray image.", 
        x.image
      ), 
      connection_id => '<<connection>>',
      model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>
)
),
sick_text AS (
  SELECT s.patient_id
  FROM two_more_modalities as s
  WHERE s.symptom_id IS NOT NULL AND AI.IF(
      FORMAT("""
      Patient sick according to the symptoms. Symptoms: %s. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.
      """, s.symptoms), 
      connection_id => '<<connection>>',
      model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>)
)
SELECT * FROM (
  SELECT t.patient_id, t.age, t.symptom_id, t.xray_id, t.bell_audio_id, 
  IF(a.patient_id IS NOT NULL, 1, IF(t.bell_audio_id IS NOT NULL, 0, NULL)) AS is_sick_audio, 
  IF(s.patient_id IS NOT NULL, 1, IF(t.symptom_id IS NOT NULL, 0, NULL)) AS is_sick_text, 
  IF(x.patient_id IS NOT NULL, 1, IF(t.xray_id IS NOT NULL, 0, NULL))  AS is_sick_image
  FROM two_more_modalities t
  LEFT JOIN sick_audio a ON t.patient_id = a.patient_id
  LEFT JOIN sick_text s ON t.patient_id = s.patient_id
  LEFT JOIN sick_image x ON t.patient_id = x.patient_id)
WHERE (is_sick_audio = 1 OR is_sick_text = 1 OR is_sick_image = 1)
AND
(is_sick_audio = 0 OR is_sick_text = 0 OR is_sick_image = 0);

