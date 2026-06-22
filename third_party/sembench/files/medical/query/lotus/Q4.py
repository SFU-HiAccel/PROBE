import os
import pandas as pd

def run(data_dir: str, scale_factor: int = 11112):
    # Load data
    symptoms_text = pd.read_csv(os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"))
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv"))

    tmp_join = patients.join(symptoms_text.set_index('patient_id'), on='patient_id', how='inner')

    # Filter data
    tmp_join = tmp_join.sem_filter('This patient has symptoms of a skin acne. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Symptoms: {symptoms}')
    avg_age = tmp_join['age'].mean()
    return pd.DataFrame({"average_acne_age": [avg_age]})
