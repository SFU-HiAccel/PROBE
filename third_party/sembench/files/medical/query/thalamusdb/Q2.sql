select distinct patients.patient_id from patients, lung_audio
where patients.patient_id = lung_audio.patient_id and patients.smoking_history != 'Current' and 
NLfilter(lung_audio.path, 'This audio recording of human lungs captures healthy lungs, without diseases.')