SELECT age, gender, smoking_history, did_family_have_cancer, patient_id 
FROM patients WHERE is_sick = TRUE 
                AND (text_diagnosis IN ('normal', '00_normal') 
                OR audio_diagnosis IN ('normal', '00_normal') 
                OR x_ray_diagnosis IN ('normal', '00_normal')) 
ORDER BY age ASC 
LIMIT 1;