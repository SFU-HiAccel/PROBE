import os
import pandas as pd
import palimpzest as pz

from palimpzest.core.lib.schemas import ImageFilepath

# define the schema for the data you read from the file
x_ray_data_cols = [
    {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the X-ray image"},
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient"},
    {"name": "xray_id", "type": int, "desc": "The integer id for the X-ray image"},
    {"name": "did_family_have_cancer", "type": int, "desc": "The integer did_family_have_cancer for family members having cancer"},
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
    x_rays = pd.read_csv(os.path.join(data_dir, "data/image_x_ray_data.csv" if scale_factor == 11112 else f"data/image_x_ray_data_{scale_factor}.csv")) 
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv")) 
    
    # Join before since PZ does not support joins
    tmp_join = patients.join(x_rays.set_index('patient_id'), on='patient_id', how='inner')[['image_path', 'patient_id', 'xray_id', 'did_family_have_cancer']]

    tmp_join = MyDataset(id="my-xray-data", x_ray_df=tmp_join)

    # Filter did_family_have_cancer
    tmp_join = tmp_join.filter(lambda row: int(row['did_family_have_cancer']) == 1)

    # Filter sickness
    tmp_join = tmp_join.sem_filter('You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if there are lung problems (considered sick/disease) according to the X-ray image.', depends_on=['image_path'])
    tmp_join = tmp_join.project(['patient_id'])
    candidates = tmp_join.limit(5)

    output = candidates.run(pz_config)
    return output
