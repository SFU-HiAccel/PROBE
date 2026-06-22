select patients.patient_id from patients, symptoms_texts 
where patients.patient_id=symptoms_texts.patient_id and 
NLfilter(symptoms_texts.symptoms, 'Patient has an allergy.')