select age, gender, smoking_history, did_family_have_cancer, patients.patient_id 
from patients, x_ray_images
where patients.patient_id = x_ray_images.patient_id and patients.did_family_have_cancer = 1 and NLfilter(x_ray_images.image_path,
    'This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image.')
limit 5