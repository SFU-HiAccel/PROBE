import os
import pandas as pd
import palimpzest as pz

def run(pz_config, data_dir: str, scale_factor: int = 11112):
    # Load data
    symptoms_text = pd.read_csv(os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"))
    symptoms_text = pz.MemoryDataset(id="symptoms", vals=symptoms_text)

    # Filter data
    symptoms_text = symptoms_text.sem_map(
        cols=[{'name': 'text_diagnosis', 'type': str, 'desc': 'Classify symptoms to one of given diseases from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Answer only one of given diseases, nothing more. Diseases: VARICOSE VEINS, DRUG REACTION, DIABETES, MALARIA, URINARY TRACT INFECTION, IMPETIGO, ACNE, HYPERTENSION, PEPTIC ULCER DISEASE, CHICKEN POX, TYPHOID, DENGUE, PNEUMONIA, MIGRAINE, GASTROESOPHAGEAL REFLUX DISEASE, PSORIASIS, COMMON COLD, CERVICAL SPONDYLOSIS, FUNGAL INFECTION, ARTHRITIS, ALLERGY, BRONCHIAL ASTHMA, JAUNDICE, DIMORPHIC HEMORRHOID.'}], 
        depends_on=['symptoms'])
    symptoms_text = symptoms_text.project(['patient_id', 'text_diagnosis'])
    
    output = symptoms_text.run(pz_config)
    return output
