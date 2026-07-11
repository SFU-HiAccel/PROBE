INSTALL flockmtl FROM community;
LOAD flockmtl;

CREATE SECRET (TYPE OPENAI, API_KEY 'my-key');  

CREATE MODEL(
   'gpt4o',
   'gpt-4o', 
   'openai', 
   {"tuple_format": "json", "batch_size": 32, "model_parameters": {"temperature": 0.7}}
);

CREATE TABLE patients AS SELECT * FROM read_csv_auto('files/medical/data/patient_data.csv');
CREATE TABLE symptoms_texts AS SELECT * FROM read_csv_auto('files/medical/data/text_symptoms_data.csv');
CREATE TABLE lung_audio AS SELECT * FROM read_csv_auto('files/medical/data/audio_lung_data.csv');
CREATE TABLE x_ray_image AS SELECT * FROM read_csv_auto('files/medical/data/image_x_ray_data.csv');
