SELECT smoking_history, COUNT(*) AS count 
FROM
(
    SELECT DISTINCT patients.patient_id, patients.smoking_history
    FROM patients, lung_audio, x_ray_images
    WHERE patients.patient_id = x_ray_images.patient_id AND patients.patient_id = lung_audio.patient_id AND patients.smoking_history = 'Current' AND
        NLfilter(lung_audio.path,'This human lung audio recording captures an audio of sick lungs, with diseases.') AND
        NLfilter(x_ray_images.image_path,'This X-ray image of human lungs shows that there are lung problems (considered sick/disease).')
)
GROUP BY smoking_history