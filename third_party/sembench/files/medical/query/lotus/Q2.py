import os
import pandas as pd
from lotus.dtype_extensions import ImageArray

def run(data_dir: str, scale_factor: int = 11112):
    raise "Audio data is not supported by Lotus"

    # # Load data
    # patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv"))
    # audio = pd.read_csv(os.path.join(data_dir, "data/audio_lung_data.csv"))

    # # Filter by smoking history and join on patient_id
    # patients = patients[patients["smoking_history"] != "Current"]
    # patient_audio_filtered = patients.join(audio.set_index('patient_id'), on='patient_id', how='inner')[["patient_id", "path"]]

    # # Wrap the image path column in ImageArray 
    # patient_audio_filtered.loc[:, "Audio"] = ImageArray(patient_audio_filtered["path"])

    # # Semantic filtering
    # filter_instruction = "You are given an audio recording of human lungs. Return true if the recording is healthy lungs, without diseases. {Audio}"
    # patient_audio_filtered = patient_audio_filtered.sem_filter(filter_instruction)

    # return patient_audio_filtered['patient_id']