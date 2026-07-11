# Medical Scenario

*Data Modalities: text, images, audio, and tables.*

## Overview

This scenario simulates an Electronic Health Records (EHR) dataset that contains comprehensive information about patients, including their medical histories, reported symptoms, and diagnostic data. The diagnostic component includes three possible modalities per patient: lung sound recordings (audio), chest X-rays (images), and descriptions of symptoms (text). Each patient may have data in up to three of these modalities.
In some cases, multiple modalities reflect the same underlying disease, whereas in others, they may indicate distinct symptoms or co-occurring conditions. This scenario explores questions concerning the presence and co-occurrence of diseases that are identifiable through multimodal data-specifically images, audio, and text-using language models for analysis and inference.

## Schema

The schema has four tables:
```
Patient(patient_id, age, gender, smoking_history, did_family_have_cancer)
SymptomsText(patient_id, symptom_id, symptom)
XrayImage(patient_id, xray_id, xray_image)
SkinCancer(patient_id, skin_image_id, skin_image)
LungAudio(patient_id,audio_id,location,filtration_type,audio)

```

The ground truth and corresponding labels are provided in the file `files/medical/data/patient_data_with_labels.csv`.
This file is not designed for multimodal data processing; instead, it serves to simplify ground truth generation through conventional SQL queries.

The `Patient` table stores demographic and background information, including `gender` (Male/Female), `age`, `smoking_history` (Former/Never/Current), and a binary field `did_family_have_cancer`, which indicates whether any family members of the patient have had cancer (1 for yes, 0 for no). Patients have 0, 1, or several deseases but only one per modality.
Furthermore, images, text, or audio can indicate that patient is healthy.

The `SymptomsText` table contains `symptom`s described by patients with `patient_id`. Each symptom is assigned to only one patient.

The `XrayImage` table contains path to chest X-Ray images, `xray_image`, each image has a `x_ray_id` and is assigned to a patient with `patient_id`. Each image is assigned to only one patient.

The `SkinMolesImage` table contains path to skin mole image, `skin_image`, each image has a `sckin_cancer_id` and is assigned to a patient with `patient_id`. Each image is assigned to only one patient.


The `LungAudio` table contains the paths of audio files (`audio` column) as well as chest zone where audio was collected (`location`).
A single patient has three audio types (`filtration_type` column) stored in the `LungAudio` table. 
Each audio has an identification `audio_id` and belongs to a patient (`patient_id` column).

## Queries

- Which patients have symptoms of allergy in our database?

- Find patients with available audio sound who are non-smokers (never smoked or stopped smoking) and healthy considering the sound recordings in our database.

- Find five patients who have cancer in their family history and have lung problems according to their X-ray images (considered sick).

- What is an average age of patients with acne?

- Count patients who are actively smoking ("Current") and whose audio and X-Ray image suggest lung diseases.

- Find patients who are sick according to at least one modality but healthy according to another modality, consider only patients with two or more available modalities. Consider only symptoms, x-ray images, and lung audio recordings.

- Find patients with any sickness in our database (consider all symptoms, x-ray images, skin moles images, and lung audio recordings)?

- Find hundred patients with malignant skin moles and cancer in the family history.

- Find patients who are sick according to both skin moles image and lung X-Ray image (semantic filter/join with two images in one prompt).

- For all symptoms, generate disease name from a given list of diseases.


## Ground Truth

- `SELECT age, gender, smoking_history, did_family_have_cancer, patient_id FROM patients WHERE text_diagnosis = 'allergy';`

- `SELECT age, gender, smoking_history, did_family_have_cancer, patient_id FROM  WHERE smoking_history != 'Current' AND audio_diagnosis = 'normal';`

- `SELECT age, gender, smoking_history, did_family_have_cancer, patient_id FROM patients WHERE did_family_have_cancer = 1 AND x_ray_diagnosis NOT IN ('00_normal', 'none') LIMIT 5;`

- `SELECT AVG(age) AS average_acne_age FROM patients WHERE text_diagnosis = 'acne';`

- `SELECT smoking_history, COUNT(*) AS count FROM patients WHERE audio_diagnosis NOT IN ('none', 'normal') AND x_ray_diagnosis NOT IN ('none', '00_normal') GROUP BY smoking_history ORDER BY count DESC;`

- `SELECT age, gender, smoking_history, did_family_have_cancer, patient_id FROM patients WHERE is_sick = TRUE AND (text_diagnosis IN ('normal', '00_normal') OR audio_diagnosis IN ('normal', '00_normal') OR x_ray_diagnosis IN ('normal', '00_normal')) ORDER BY age ASC LIMIT 1;`

- `SELECT age, gender, smoking_history, did_family_have_cancer, patient_id FROM patient_df WHERE is_sick = 1`


## Query Summary
Here's a summary table of the features for each query:

| **Query** | **Data Types** | | | **Join Operations** | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **Table** | **Text** | **Image** | **Audio** | **# Joins** | **Text Join** | **Image Join** | **Audio Join** | **Text-Image Join** |
|**Q1**       | ✅ | ✅       | ❌        | ❌          | 1          | ✅             | ❌             | ❌             | ❌  |
|**Q2**       | ✅ | ❌       | ❌        | ✅          | 1          | ❌             | ❌             | ✅             | ❌  |
|**Q3**       | ✅ | ❌       | ✅        | ❌          | 1          | ❌             | ✅             | ❌             | ❌  |
|**Q4**       | ✅ | ✅       | ❌        | ❌          | 1          | ✅             | ❌             | ❌             | ❌  |
|**Q5**       | ✅ | ❌       | ✅        | ✅          | 2          | ❌             | ✅             | ✅             | ❌  |
|**Q6**       | ✅ | ✅       | ✅        | ✅          | 9          | ✅             | ✅             | ✅             | ❌  |
|**Q7**       | ✅ | ✅       | ✅        | ✅          | 5          | ✅             | ✅             | ✅             | ❌  |
|**Q8**       | ✅ | ❌       | ✅        | ❌          | 4          | ❌             | ✅             | ❌             | ❌  |
|**Q9**       | ✅ | ❌       | ✅        | ❌          | 4          | ❌             | ✅             | ❌             | ❌  |
|**Q10**      | ✅ | ✅       | ❌        | ❌          | 4          | ❌             | ❌             | ❌             | ❌  |
---

## Supported operators

|                       |**Q1**| **Q2**        | **Q3**        | **Q4**        | **Q5**        | **Q6**              | **Q7**                | **Q8**        | **Q9** | **Q10**         |
|-----------------------|------|--------------|---------------|---------------|--------------|----------------------|------------------------|---------------|----------------------------|----------------------|
| **Table**             | ✅   | ✅            | ✅            | ✅            | ✅            | ✅                   | ✅                      | ✅            | ✅     | ✅       |
| **Text**              | ✅   | ❌            | ❌            | ✅            | ❌            | ✅                   | ✅                      | ❌            | ❌     | ✅       |
| **Image**             | ❌   | ❌            | ✅            | ❌            | ✅            | ✅                   | ✅                      | ✅            | ✅     | ❌       |
| **Audio**             | ❌   | ✅            | ❌            | ❌            | ✅            | ✅                   | ✅                      | ❌            | ❌     | ❌       |
| **BigQuery**          | ✅   | ✅            | ✅            | ✅            | ✅            | ✅                   | ✅                      | ✅            | ✅     | ✅       |
| **Palimpzest**         | ✅   | ❌            | ✅            | ✅            | ❌            | ❌                   | ❌                      | ✅            | ✅     | ✅       |
| **Lotus**             | ✅   | ❌            | ✅            | ✅            | ❌            | ❌                   | ❌                      | ✅            | ✅     | ✅       |
| **ThalamusDB**        | ✅   | ✅            | ✅            | ✅            | ✅            | ❓                   | ✅                      | ✅            | ✅     | ❌       |
| **FlockMTL**          | ✅   | ❌            | ❌            | ✅            | ❌            | ❌                   | ❌                      | ❌            | ❌     | ✅       |
| **Operators**         | AI.IF<br>+text    | AI.IF<br>+audio   | AI.IF<br>+image   | AI.IF<br>+text    | AI.IF<br>+text    | AI.IF text<br>+audio     | AI.IF for<br>each modality | AI.IF<br>+image   | AI.IF<br>+2 images in<br>1 prompt | AI.GENERATE<br>+text    |
