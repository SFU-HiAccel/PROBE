import os
import pandas as pd
from lotus.dtype_extensions import ImageArray

def run(data_dir: str, scale_factor: int = 11112):
    # Load data
    x_rays = pd.read_csv(os.path.join(data_dir, "data/image_x_ray_data.csv" if scale_factor == 11112 else f"data/image_x_ray_data_{scale_factor}.csv"))
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv"))
    
    # Filter and join
    patients = patients[patients["did_family_have_cancer"] == 1]
    tmp_join = patients.join(x_rays.set_index('patient_id'), on='patient_id', how='inner')[['image_path', 'patient_id', 'xray_id', 'did_family_have_cancer']]
    
    tmp_join.loc[:, "Image"] = ImageArray(tmp_join["image_path"])

    # Filter sickness
    filter_instruction = "You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if there are lung problems (considered sick/disease) according to the X-ray image. {Image}"
    tmp_join = tmp_join.sem_filter(filter_instruction)
    
    candidates = tmp_join.head(5)

    return candidates["patient_id"]
