import os
import pandas as pd
import palimpzest as pz

from palimpzest.core.lib.schemas import ImageFilepath

# define the schema for the data you read from the file
x_ray_data_cols = [
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient"},
    {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the skin image"},
    {"name": "image_path_xray", "type": ImageFilepath, "desc": "The filepath containing the X-ray image"},
]


class MyDataset(pz.IterDataset):
  def __init__(self, id: str, x_ray_df: pd.DataFrame):
    super().__init__(id=id, schema=x_ray_data_cols)
    self.x_ray_df = x_ray_df

  def __len__(self):
    return len(self.x_ray_df)

  def __getitem__(self, idx: int):
    # get row from dataframe
    return self.x_ray_df.iloc[idx].to_dict()
  

def run(pz_config, data_dir: str, scale_factor: int = 11112):
    # Load data
    cancer_data = pd.read_csv(os.path.join(data_dir, "data/image_skin_data.csv" if scale_factor == 11112 else f"data/image_skin_data_{scale_factor}.csv")) 
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv")) 
    x_rays = pd.read_csv(os.path.join(data_dir, "data/image_x_ray_data.csv" if scale_factor == 11112 else f"data/image_x_ray_data_{scale_factor}.csv")) 
    
    # Join before since PZ does not support joins
    tmp_join = patients.join(cancer_data.set_index('patient_id'), on='patient_id', how='inner')[['image_path', 'patient_id']]
    tmp_join = tmp_join.join(x_rays.set_index('patient_id'), on='patient_id', how='inner')[['patient_id', 'image_path', 'image_path_xray']]

    tmp_join = MyDataset(id="my-skin-data", x_ray_df=tmp_join)

    # Filter sickness
    tmp_join = tmp_join.sem_filter("You are given an image of a human skin mole and an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true only if both images indicate any diseases.", depends_on=['image_path', "image_path_xray"])
    tmp_join = tmp_join.project(['patient_id'])

    output = tmp_join.run(pz_config)
    return output
