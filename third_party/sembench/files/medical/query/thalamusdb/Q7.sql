WITH sick_audio AS (
    select distinct patients.patient_id from patients, lung_audio
            where patients.patient_id = lung_audio.patient_id and patients.smoking_history != 'Current' and 
            NLfilter(lung_audio.path, 'This audio recording of human lungs captures sick lungs, with diseases.')
),
sick_text AS (
    select patients.patient_id from patients, symptoms_texts 
            where patients.patient_id=symptoms_texts.patient_id and 
            NLfilter(symptoms_texts.symptoms, 'This patient is sick.')
),
sick_image AS(
    select age, gender, smoking_history, did_family_have_cancer, patients.patient_id 
    from patients, x_ray_images
    where patients.patient_id = x_ray_images.patient_id and NLfilter(x_ray_images.image_path,
            'This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image.')
),
sick_cancer AS(
    select age, gender, smoking_history, did_family_have_cancer, patients.patient_id 
    from patients, skin_images
    where patients.patient_id = skin_images.patient_id and NLfilter(skin_images.image_path,
                'This image shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image.')
)

SELECT sick_audio.patient_id FROM sick_audio
UNION DISTINCT
SELECT sick_text.patient_id FROM sick_text
UNION DISTINCT
SELECT sick_image.patient_id FROM sick_image
UNION DISTINCT
SELECT sick_cancer.patient_id FROM sick_cancer;