import os
import pandas as pd
import palimpzest as pz

from palimpzest.core.lib.schemas import ImageFilepath, AudioFilepath
import copy


data_audio_cols = [
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient"},
    {"name": "path", "type": AudioFilepath, "desc": "The filepath containing audios."}
]

data_xray_cols = [
    {"name": "xray_id", "type": int, "desc": "The integer id for the X-ray image"},
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient"},
    {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the X-ray image"},
]

data_moles_cols = [
    {"name": "skin_image_id", "type": int, "desc": "The integer id for the X-ray image"},
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient"},
    {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the X-ray image"},
]


class MyDataset(pz.IterDataset):
  def __init__(self, id: str, x_ray_df: pd.DataFrame, schema: list):
    super().__init__(id=id, schema=schema)
    self.x_ray_df = x_ray_df

  def __len__(self):
    return len(self.x_ray_df)

  def __getitem__(self, idx: int):
    # get row from dataframe
    return self.x_ray_df.iloc[idx].to_dict()
  

def run(pz_config, data_dir: str, scale_factor: int = 11112):
    # Load data
    # patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv")) 
    x_rays = pd.read_csv(os.path.join(data_dir, "data/image_x_ray_data.csv" if scale_factor == 11112 else f"data/image_x_ray_data_{scale_factor}.csv"))
    audio = pd.read_csv(os.path.join(data_dir, "data/audio_lung_data.csv" if scale_factor == 11112 else f"data/audio_lung_data_{scale_factor}.csv"))
    text = pd.read_csv(os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"))
    skin_moles = pd.read_csv(os.path.join(data_dir, "data/image_skin_data.csv" if scale_factor == 11112 else f"data/image_skin_data_{scale_factor}.csv")) 

    audio_pz = MyDataset(id="audio", x_ray_df=audio, schema=data_audio_cols)
    audio_pz = audio_pz.sem_filter("You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if the patient is sick according to the audio.", depends_on=["path"])
    output_audio = audio_pz.run(copy.deepcopy(pz_config))
    audio_cost = output_audio.execution_stats.total_execution_cost
    output_audio = output_audio.to_df()
    
    x_rays_pz = MyDataset(id="x_rays", x_ray_df=x_rays, schema=data_xray_cols)
    x_rays_pz = x_rays_pz.sem_filter("You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if there are lung problems (considered sick/disease) according to the X-ray image.", depends_on=["image_path"])
    output_xrays = x_rays_pz.run(copy.deepcopy(pz_config))
    xray_cost = output_xrays.execution_stats.total_execution_cost
    output_xrays = output_xrays.to_df()

    text_pz = pz.MemoryDataset(id="symptoms", vals=text)
    text_pz = text_pz.sem_filter("This patient is sick according to the symptoms. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.", depends_on=["symptoms"])
    output_text = text_pz.run(copy.deepcopy(pz_config))
    text_cost = output_text.execution_stats.total_execution_cost
    output_text = output_text.to_df()
    
    moles_pz = MyDataset(id="moles", x_ray_df=skin_moles, schema=data_moles_cols)
    moles_pz = moles_pz.sem_filter("You are given an image of human skin mole from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if mole or skin area is malignant (considered abnormal/cancerous/sick) according to the image.", depends_on=["image_path"])
    output_moles = moles_pz.run(copy.deepcopy(pz_config))
    moles_cost = output_moles.execution_stats.total_execution_cost
    output_moles = output_moles.to_df()

    res = pd.concat([output_audio[['patient_id']], output_text[['patient_id']], output_moles[['patient_id']], output_xrays[['patient_id']]], ignore_index=True).drop_duplicates()

    cost = audio_cost + text_cost + xray_cost + moles_cost

    return (res, cost)