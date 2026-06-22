import os
import numpy as np
import pandas as pd
import palimpzest as pz

from palimpzest.core.lib.schemas import ImageFilepath, AudioFilepath
import copy
from palimpzest.constants import Model

data_audio_cols = [
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient"},
    {"name": "path_e", "type": AudioFilepath, "desc": "The filepath containing audios."},
    {"name": "path_b", "type": AudioFilepath, "desc": "The filepath containing audios."},
    {"name": "path_d", "type": AudioFilepath, "desc": "The filepath containing audios."},
]

data_xray_cols = [
    {"name": "xray_id", "type": int, "desc": "The integer id for the X-ray image"},
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient"},
    {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the X-ray image."},
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
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv")) 
    x_rays = pd.read_csv(os.path.join(data_dir, "data/image_x_ray_data.csv" if scale_factor == 11112 else f"data/image_x_ray_data_{scale_factor}.csv"))
    audio = pd.read_csv(os.path.join(data_dir, "data/audio_lung_data.csv" if scale_factor == 11112 else f"data/audio_lung_data_{scale_factor}.csv"))
    text = pd.read_csv(os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"))

    # Unpivot audio. Map filtration types to output columns
    type_to_col = {"bell": "path_b", "extended": "path_e", "diaphragm": "path_d"}
    audio["wide_col"] = audio["filtration_type"].map(type_to_col)

    # Pivot to one row per patient+location
    audio = (
        audio.pivot_table(
            index=["patient_id", "location"],
            columns="wide_col",
            values="path",
            aggfunc="first",
        )
        .reset_index()
    )
    audio.columns.name = None

    # Ensure all expected columns exist (even if some types are missing)
    for c in ["path_b", "path_e", "path_d"]:
        if c not in audio.columns:
            audio[c] = pd.NA

    # Join 
    tmp_join = patients.join(x_rays.set_index('patient_id'), on='patient_id', how='outer')[['image_path', 'patient_id', 'xray_id', 'smoking_history']]
    tmp_join = tmp_join.join(audio.set_index('patient_id'), on='patient_id', how='outer')[['patient_id', 'smoking_history', 'xray_id', 'image_path', 'path_e', 'path_b', 'path_d']]
    tmp_join = tmp_join.join(text.set_index('patient_id'), on='patient_id', how='outer')[['patient_id', 'smoking_history', 'xray_id', 'image_path', 'path_e', 'path_b', 'path_d', 'symptom_id', 'symptoms']]

    tmp_join["non_missing_count"] = tmp_join[["xray_id", "symptom_id", "path_e"]].notna().sum(axis=1)

    # Filter: keep rows with at least 2 present
    tmp_join = tmp_join[tmp_join["non_missing_count"] >= 2]
    tmp_join = tmp_join.drop(columns=["non_missing_count"])

    tmp_join = tmp_join.sample(100)

    # Filter each modality separately
    audio = audio.loc[audio["patient_id"].isin(tmp_join["patient_id"].tolist()), ["patient_id", "path_b", "path_e", "path_d"]]
    x_rays = x_rays[x_rays["patient_id"].isin(tmp_join["patient_id"].tolist())]
    text = text[text["patient_id"].isin(tmp_join["patient_id"].tolist())]

    audio_pz = MyDataset(id="audio", x_ray_df=audio, schema=data_audio_cols)
    audio_pz = audio_pz.sem_map(cols=[{'name': 'e_diagnosis', 'type': str, 'desc': "You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return TRUE if the recording captures sick lungs, with diseases. If healthy return FALSE. Answer only TRUE or FALSE."}], depends_on=['path_e'])
    audio_pz = audio_pz.sem_map(cols=[{'name': 'b_diagnosis', 'type': str, 'desc': "You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return TRUE if the recording captures sick lungs, with diseases. If healthy return FALSE. Answer only TRUE or FALSE."}], depends_on=['path_b'])
    audio_pz = audio_pz.sem_map(cols=[{'name': 'd_diagnosis', 'type': str, 'desc': "You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return TRUE if the recording captures sick lungs, with diseases. If healthy return FALSE. Answer only TRUE or FALSE."}], depends_on=['path_d'])

    # FIXME filter with three audio is not supported
    # audio_pz = audio_pz.sem_filter("You are given three audio recordings of human lungs. Return true if the recording captures sick lungs, with diseases.", depends_on=["path_e"])
    
    output_audio = audio_pz.run(copy.deepcopy(pz_config))
    audio_cost = output_audio.execution_stats.total_execution_cost
    output_audio = output_audio.to_df()
    output_audio["e_diagnosis"] = output_audio["e_diagnosis"].astype(str).str.lower()
    output_audio["b_diagnosis"] = output_audio["b_diagnosis"].astype(str).str.lower()
    output_audio["d_diagnosis"] = output_audio["d_diagnosis"].astype(str).str.lower()
    output_audio = output_audio[(output_audio["e_diagnosis"] == "true") | (output_audio["b_diagnosis"] == "true") | (output_audio["d_diagnosis"] == "true")]
    output_audio["sick_audio"] = 1
    
    x_rays_pz = MyDataset(id="x_rays", x_ray_df=x_rays, schema=data_xray_cols)
    x_rays_pz = x_rays_pz.sem_filter("You are given an X-ray image of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if there are lung problems (considered sick/disease) according to the X-ray image.", depends_on=["image_path"])
    output_xrays = x_rays_pz.run(copy.deepcopy(pz_config))
    xray_cost = output_xrays.execution_stats.total_execution_cost
    output_xrays = output_xrays.to_df()
    output_xrays["sick_xrays"] = 1

    text_pz = pz.MemoryDataset(id="symptoms", vals=text)
    text_pz = text_pz.sem_filter("This patient is sick according to the symptoms. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.", depends_on=["symptoms"])
    output_text = text_pz.run(copy.deepcopy(pz_config))
    text_cost = output_text.execution_stats.total_execution_cost
    output_text = output_text.to_df()
    output_text["sick_text"] = 1

    tmp_join.reset_index(inplace=True)

    res = tmp_join.join(output_audio.set_index('patient_id'), on='patient_id', how='outer', rsuffix="_au")
    res = res.join(output_xrays.set_index('patient_id'), on='patient_id', how='outer', rsuffix="_xr")
    res = res.join(output_text.set_index('patient_id'), on='patient_id', how='outer', rsuffix="_txt")

    # res["sick_text"] = 0 if (res["symptoms_txt"].isna()) and (res["sick_text"].isna()) else res["sick_text"]
    # res["sick_xrays"] = 0 if (res["image_path_xr"].isna()) and (res["sick_xrays"].isna()) else res["sick_xrays"]
    # res["sick_audio"] = 0 if (res["path_e"].isna()) and (res["sick_audio"].isna()) else res["sick_audio"]

    res["sick_text"] = np.where(res["symptoms_txt"].isna() & res["sick_text"].isna(), 0, res["sick_text"])
    res["sick_xrays"] = np.where(res["image_path_xr"].isna() & res["sick_xrays"].isna(), 0, res["sick_xrays"])
    res["sick_audio"] = np.where(res["path_e"].isna() & res["sick_audio"].isna(), 0, res["sick_audio"])

    # Find sick and helthy according to two different modalities
    res = res[((
       (res["sick_audio"] == 1) | (res["sick_text"] == 1) | (res["sick_xrays"] == 1)) & 
       ((res["sick_audio"] == 0) | (res["sick_text"] == 0) | (res["sick_xrays"] == 0)
    ))]

    cost = audio_cost + text_cost + xray_cost

    return (res[["patient_id"]], cost)

