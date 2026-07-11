import os
import pandas as pd
import palimpzest as pz

def run(pz_config, data_dir: str, scale_factor: int = 11112):
    # Load data
    symptoms_text = pd.read_csv(os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"))
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv"))

    # Join before since PZ does not support joins
    tmp_join = patients.join(symptoms_text.set_index('patient_id'), on='patient_id', how='inner')
    tmp_join = pz.MemoryDataset(id="symptoms_patients", vals=tmp_join)

    # Filter and get avg age
    tmp_join = tmp_join.sem_filter('This patient has symptoms of a skin acne. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.', depends_on=['symptoms'])
    tmp_join = tmp_join.project(['age'])
    tmp_join = tmp_join.average()

    output = tmp_join.run(pz_config)
    
    return output
