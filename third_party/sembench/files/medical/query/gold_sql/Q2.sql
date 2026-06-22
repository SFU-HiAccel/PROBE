SELECT age, gender, smoking_history, did_family_have_cancer, patient_id FROM patients WHERE smoking_history != 'Current' AND audio_diagnosis = 'normal';
