SELECT AVG(patients.age)
FROM patients 
JOIN symptoms_texts 
WHERE patients.patient_id = symptoms_texts.patient_id AND NLfilter(symptoms_texts.symptoms, 'Patient has a skin acne.')