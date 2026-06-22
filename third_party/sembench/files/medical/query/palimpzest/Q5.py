import os
import pandas as pd
import palimpzest as pz

from palimpzest.core.lib.schemas import ImageFilepath, AudioFilepath
from palimpzest.core.elements.groupbysig import GroupBySig


data_cols = [
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient"},
    {"name": "smoking_history", "type": str, "desc": "The string smoking_history."},
    {"name": "xray_id", "type": int, "desc": "The integer id for the X-ray image"},
    {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the X-ray image"},
    {"name": "audio_id", "type": int, "desc": "The integer id for the audio"},
    {"name": "path", "type": AudioFilepath, "desc": "The filepath containing audios."},
]


class MyDataset(pz.IterDataset):
  def __init__(self, id: str, x_ray_df: pd.DataFrame):
    super().__init__(id=id, schema=data_cols)
    self.x_ray_df = x_ray_df

  def __len__(self):
    return len(self.x_ray_df)

  def __getitem__(self, idx: int):
    # get row from dataframe
    return self.x_ray_df.iloc[idx].to_dict()
  

def run(pz_config, data_dir: str, scale_factor: int = 11112):
    # Load data
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv")) 
    x_rays = pd.read_csv(os.path.join(data_dir, "data/image_x_ray_data.csv" if scale_factor == 11112 else f"data/image_x_ray_data_{scale_factor}.csv"))
    audio = pd.read_csv(os.path.join(data_dir, "data/audio_lung_data.csv" if scale_factor == 11112 else f"data/audio_lung_data_{scale_factor}.csv"))
    
    # Join 
    tmp_join = patients.join(x_rays.set_index('patient_id'), on='patient_id', how='inner')[['image_path', 'patient_id', 'xray_id', 'smoking_history']]
    tmp_join = tmp_join.join(audio.set_index('patient_id'), on='patient_id', how='inner')[['patient_id', 'smoking_history', 'xray_id', 'image_path', 'path', 'audio_id']]
    
    tmp_join = MyDataset(id="my-data", x_ray_df=tmp_join)

    # Filter did_family_have_cancer
    tmp_join = tmp_join.filter(lambda row: row['smoking_history'] == 'Current')

    # Filter sickness
    tmp_join = tmp_join.sem_filter('You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if the recording captures an audio of sick lungs, with diseases.', depends_on=['path'])
    tmp_join = tmp_join.sem_filter("You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if there are lung problems (considered sick/disease) according to the X-ray image.", depends_on=['image_path'])
    tmp_join = tmp_join.project(['patient_id', 'smoking_history'])
    tmp_join = tmp_join.distinct(distinct_cols=['patient_id', 'smoking_history'])

    # Group by
    gby_desc = GroupBySig(group_by_fields=["smoking_history"], agg_funcs=["count"], agg_fields=["smoking_history"])
    res = tmp_join.groupby(gby_desc)

    output = res.run(pz_config)
    return output
