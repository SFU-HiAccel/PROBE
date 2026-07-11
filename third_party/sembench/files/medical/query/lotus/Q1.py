import os
import pandas as pd

def run(data_dir: str, scale_factor: int = 11112):
    # Load data
    symptoms_text = pd.read_csv(os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"))

    # Filter data
    symptoms_text = symptoms_text.sem_filter('This patient has symptoms of an allergy. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Symptoms: {symptoms}')
    return symptoms_text['patient_id']