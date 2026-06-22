import os
import pandas as pd
from lotus.dtype_extensions import ImageArray

def run(data_dir: str, scale_factor: int = 11112):
    # Load data
    cancer_data = pd.read_csv(os.path.join(data_dir, "data/image_skin_data.csv" if scale_factor == 11112 else f"data/image_skin_data_{scale_factor}.csv")) 
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv")) 
    x_rays = pd.read_csv(os.path.join(data_dir, "data/image_x_ray_data.csv" if scale_factor == 11112 else f"data/image_x_ray_data_{scale_factor}.csv")) 
    x_rays.rename(columns={"image_path": "image_path_xray"}, inplace=True)
    
    # Join before since PZ does not support joins
    tmp_join = patients.join(cancer_data.set_index('patient_id'), on='patient_id', how='inner')[['image_path', 'patient_id']]
    tmp_join = tmp_join.join(x_rays.set_index('patient_id'), on='patient_id', how='inner')[['patient_id', 'image_path', 'image_path_xray']]

    tmp_join.loc[:, "Image"] = ImageArray(tmp_join["image_path"])
    tmp_join.loc[:, "ImageX"] = ImageArray(tmp_join["image_path_xray"])

    # Filter sickness
    filter_instruction = "You are given an image of a human skin mole and an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true only if both images indicate any diseases. {Image} {ImageX}"
    tmp_join = tmp_join.sem_filter(filter_instruction)

    return tmp_join["patient_id"]
