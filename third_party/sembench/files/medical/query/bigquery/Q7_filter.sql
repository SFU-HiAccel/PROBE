WITH sick_audio AS (
  SELECT DISTINCT p.patient_id 
  FROM medical_dataset.patients  AS p
  JOIN medical_dataset.audio_mm  AS a 
  ON p.patient_id = a.patient_id
  WHERE AI.IF(
        prompt => (
          "You are given three audio recordings of human lungs. Return true if the recording captures sick lungs, with diseases.", 
          a.image
        ), 
        connection_id => '<<connection>>',
        model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}}',
        endpoint => 'gemini-2.5-flash')),
sick_text AS (
  SELECT p.patient_id 
  FROM medical_dataset.patients  AS p
  JOIN medical_dataset.symptoms_texts  AS s
  ON p.patient_id = s.patient_id 
  WHERE AI.IF(
      FORMAT("""
      Is patient sick according to the symptoms?
      Symptoms: %s
      """, s.symptoms), 
      connection_id => '<<connection>>',
      model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}}',
      endpoint => 'gemini-2.5-flash')),
sick_image AS(
  SELECT p.patient_id
  FROM medical_dataset.x_ray_mm as x
  JOIN medical_dataset.patients  AS p
  ON p.patient_id = x.patient_id 
  WHERE AI.IF(
      prompt => (
        "You are given an X-ray image of human lungs. Return true if there are lung problems (considered sick/disease) according to the X-ray image.", 
        x.image
      ), 
      connection_id => '<<connection>>',
      model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}}',
      endpoint => 'gemini-2.5-flash')
),
sick_cancer AS(
  SELECT p.patient_id
  FROM medical_dataset.skin_cancer_mm as x
  JOIN medical_dataset.patients  AS p
  ON p.patient_id = x.patient_id 
  WHERE AI.IF(
      prompt => (
        "You are given an image of human skin mole. Return true if mole or skin area is malignant (considered abnormal/cancerous/sick) according to the image.", 
        x.image
      ), 
      connection_id => '<<connection>>',
      model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}}',
      endpoint => 'gemini-2.5-flash')
)

SELECT AVG(p.age) as age
FROM medical_dataset.patients as p, sick_audio, sick_text, sick_image, sick_cancer
WHERE p.patient_id = sick_audio.patient_id OR
  p.patient_id = sick_text.patient_id OR
  p.patient_id = sick_image.patient_id OR
  p.patient_id = sick_cancer.patient_id