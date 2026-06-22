WITH sick_audio AS(
    SELECT two_more_modalities.patient_id as patient_id
    FROM two_more_modalities
    WHERE two_more_modalities.bell_audio_id IS NOT NULL AND 
        (NLfilter(two_more_modalities.extended_audio, 'This audio recording of human lungs captures sick lungs, with diseases.') OR 
        NLfilter(two_more_modalities.bell_audio, 'This audio recording of human lungs captures sick lungs, with diseases.') OR 
        NLfilter(two_more_modalities.diaphragm_audio, 'This audio recording of human lungs captures sick lungs, with diseases.'))
),
sick_image AS(
    SELECT two_more_modalities.patient_id as patient_id
    FROM two_more_modalities
    WHERE two_more_modalities.xray_id IS NOT NULL AND NLfilter(two_more_modalities.image_path,
        'This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image.')
),
sick_text AS (
    SELECT two_more_modalities.patient_id as patient_id
    FROM two_more_modalities
    WHERE two_more_modalities.symptom_id IS NOT NULL AND NLfilter(two_more_modalities.symptoms, 'This patient is sick.')
)
SELECT * FROM (
SELECT two_more_modalities.patient_id, two_more_modalities.age, two_more_modalities.symptom_id, two_more_modalities.xray_id, two_more_modalities.bell_audio_id, 
IF(two_more_modalities.patient_id IS NOT NULL, 1, IF(two_more_modalities.bell_audio_id IS NOT NULL, 0, NULL)) AS is_sick_audio, 
IF(two_more_modalities.patient_id IS NOT NULL, 1, IF(two_more_modalities.symptom_id IS NOT NULL, 0, NULL)) AS is_sick_text, 
IF(two_more_modalities.patient_id IS NOT NULL, 1, IF(two_more_modalities.xray_id IS NOT NULL, 0, NULL))  AS is_sick_image
FROM two_more_modalities
LEFT JOIN sick_audio ON two_more_modalities.patient_id = sick_audio.patient_id
LEFT JOIN sick_text ON two_more_modalities.patient_id = sick_text.patient_id
LEFT JOIN sick_image ON two_more_modalities.patient_id = sick_image.patient_id)
WHERE (is_sick_audio = 1 OR is_sick_text = 1 OR is_sick_image = 1)
AND
(is_sick_audio = 0 OR is_sick_text = 0 OR is_sick_image = 0);