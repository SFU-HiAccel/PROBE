select age, gender, smoking_history, did_family_have_cancer, patients.patient_id 
from patients, skin_images
where patients.patient_id = skin_images.patient_id and patients.did_family_have_cancer = 1 and NLfilter(skin_images.image_path,
    'This image shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image.')
limit 100;