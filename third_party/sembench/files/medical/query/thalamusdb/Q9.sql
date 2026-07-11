SELECT patients.patient_id 
FROM patients, skin_images, x_ray_images
WHERE patients.patient_id = skin_images.patient_id AND patients.patient_id = x_ray_images.patient_id AND
    NLjoin(skin_images.image_path, x_ray_images.image_path, 'Both images indicate diseases, one image shows malignant human skin mole, and another image shows sick human lungs with diseases.');