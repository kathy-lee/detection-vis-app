
import streamlit as st
import requests
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from streamlit_extras.switch_page_button import switch_page
from streamlit_ace import st_ace



backend_service = os.getenv('BACKEND_SERVICE', 'localhost')
model_rootdir = os.getenv('MODEL_ROOTDIR')


def _load_model_features():
  """model name -> feature codes it consumes. Single source of truth for both
  the model dropdown (model_zoo) and the feature-compatibility guard."""
  with open(os.path.join('detection_vis_backend', 'networks', 'model_features.json'), 'r') as f:
    return json.load(f)

MODEL_FEATURES = _load_model_features()


def go_to_jobs_button(key):
    """Jump to the Jobs page, which is the single place to monitor all jobs."""
    if st.button("View jobs ▸", key=key):
        switch_page("jobs")

st.set_page_config(
    page_title = "Have fun ",
    page_icon = ":red_car:", 
)


if 'datafiles_chosen' not in st.session_state or 'features_chosen' not in st.session_state or not st.session_state.features_chosen:
  st.info("Please choose data and feature first.")
  st.stop()


st.write("You have chosen the following data file(s):")
s = ''
for i in st.session_state.datafiles_chosen:
  s += "- " + i["name"] + "\tfrom\t" + i["parse"] + "\n"
st.markdown(s)

edit_data_action = st.button("Edit data", key="edit_data_btn")
if edit_data_action:
  switch_page("choose data")


st.write("You have chosen the following feature(s):")
s = ''
for i in st.session_state.features_chosen:
  s += "- " + i + "\n"
st.markdown(s)
edit_feature_action = st.button("Edit feature", key="edit_feature_btn")
if edit_feature_action:
  switch_page("get features")


dataset_name = st.session_state.datafiles_chosen[0]["parse"]
if len(st.session_state.datafiles_chosen) > 1:
  if all(f["parse"] == dataset_name for f in st.session_state.datafiles_chosen):
    if dataset_name in ("RADIal", "CARRADA", "RADDetDataset", "ViewOfDelft"):
      split_mode = st.radio("Select the data split way:", ["sequence", "random", "original"], horizontal=True, key=f"radio_splitmode")
      st.write("Original mode is the default split which the dataset provides.")
    else:
      split_mode = st.radio("Select the data split way:", ["sequence", "random"], horizontal=True, key=f"radio_splitmode")
  else:
    st.info("Train with files from different datasets is not supported currently.")
    st.stop()
else:
  if dataset_name in ("RADIal", "CARRADA", "RADDetDataset", "ViewOfDelft"):
    split_mode = st.radio("Select the data split way:", ["random", "original"], horizontal=True, key=f"radio_splitmode")
  else:
    split_mode = "random"

st.session_state.split_mode = split_mode

if st.session_state.split_mode == 'random':
  if 'split_ratios' not in st.session_state:
    st.session_state.split_ratios = []
  split_value = st.slider('Select the ratio for Train/Validation/Test data:', 0.0, 1.0, (0.70, 0.90))
  st.write(f"Split ratios:   Train: {split_value[0]:.2f}, Validation: {split_value[1]-split_value[0]:.2f}, Test: {1.0-split_value[1]:.2f}")
  st.session_state.split_ratios = [split_value[0], split_value[1]-split_value[0], 1.0-split_value[1]]
elif st.session_state.split_mode == 'sequence':
  if 'datafiles_split' not in st.session_state or \
    set([item['id'] for sublist in st.session_state.datafiles_split.values() for item in sublist]) != set([item['id'] for item in st.session_state.datafiles_chosen]):
    st.session_state.datafiles_split = {key: [] for key in ["train", "val", "test"]}
  splits = ["train", "val", "test"]
  for f in st.session_state.datafiles_chosen:
    radiobox = f"split_{f['id']}"
    if radiobox not in st.session_state:
      radio_select = st.radio(f"File {f['name']} as:", ["train", "val", "test"], index=None, horizontal=True, key=f"radio_{radiobox}")
      # st.session_state[radiobox] = False
    elif st.session_state[radiobox]:
      radio_select = st.radio(f"File {f['name']} as:", ["train", "val", "test"], index=splits.index(st.session_state[radiobox]), horizontal=True, key=f"radio_{radiobox}")
    # else:
    #   radio_select = col2.radio("", ["train", "validation", "test"], horizontal=True, key=f"radio_{radiobox}")
    if radio_select and f not in st.session_state.datafiles_split[radio_select]:
      st.session_state[radiobox] = radio_select
      st.session_state.datafiles_split[radio_select].append(f)
      left_splits = [x for x in splits if x != radio_select]
      for split in left_splits:
        if f in st.session_state.datafiles_split[split]:
          st.session_state.datafiles_split[split].remove(f)
          break


def check_sequence_split():
  """Validate the per-file split assignment before submitting a sequence-mode training.

  The radios start unselected, so a file can silently belong to NO split, and nothing
  forces train/val to be non-empty -- the backend would reject both (or, before that fix,
  crash with "datasets should not be an empty iterable"). Catch it at the click instead.
  Returns True when the submission may proceed.
  """
  if st.session_state.split_mode != 'sequence':
    return True
  split = st.session_state.datafiles_split
  assigned_ids = {f['id'] for files in split.values() for f in files}
  unassigned = [f['name'] for f in st.session_state.datafiles_chosen if f['id'] not in assigned_ids]
  if unassigned:
    st.warning(f"Not assigned to any split: {', '.join(unassigned)}. "
               "Assign every chosen file to train, val or test.")
    return False
  if not split['train'] or not split['val']:
    st.warning("Assign at least one file to 'train' and one to 'val'.")
    return False
  if not split['test']:
    st.info("No test file assigned — test evaluation will be skipped.")
  return True



train_modes = ["Train from scratch", "Train on a pre-trained model"]
train_mode = st.radio("Choose a way to train model:", train_modes)

if "model_configs" not in st.session_state:
  st.session_state.model_configs = {}
if "train_configs" not in st.session_state:
  st.session_state.train_configs = {}
if "model_chosen" not in st.session_state:
  st.session_state.model_chosen = None

def config_editor(model_cfg, train_cfg):
  st.write("Model config:")
  formatted_model_configs = json.dumps(model_cfg, indent=2)
  formatted_model_configs = st_ace(value=formatted_model_configs,theme="solarized_dark",
                keybinding="vscode",
                min_lines=20,
                max_lines=None,
                font_size=14,
                tab_size=4,
                wrap=False,
                show_gutter=True,
                show_print_margin=False,
                readonly=False,
                annotations=None)
  model_configs = json.loads(formatted_model_configs)
  st.session_state.model_configs = model_configs
  
  st.write("Train config:")
  formatted_train_configs = json.dumps(train_cfg, indent=2)
  formatted_train_configs = st_ace(value=formatted_train_configs,theme="solarized_dark",
                keybinding="vscode",
                min_lines=20,
                max_lines=None,
                font_size=14,
                tab_size=4,
                wrap=False,
                show_gutter=True,
                show_print_margin=False,
                readonly=False,
                annotations=None)
  train_configs = json.loads(formatted_train_configs)
  st.session_state.train_configs = train_configs
  return 


def find_default_configs(dataset, feature, model):
  # model config
  if model in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti", "MVRECORD"):
    with open(os.path.join('detection_vis_backend', 'networks', 'RECORD_model_config.json'), 'r') as file:
      data = json.load(file)
    if model in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti"):
      if len(feature) > 1:
        st.error('RECORD network supports only single feature as input', icon="🚨")
      if dataset in data[model].keys():
        if feature[0] in data[model][dataset].keys():
          model_cfg = data[model][dataset][feature[0]]
        else:
          model_cfg = data[model][dataset]
      else:
        model_cfg = data[model]
      # if dataset == "CRUW":
      #   model_cfg = data[model][dataset]
      # elif dataset == "CARRADA":
      #   model_cfg = data[model][dataset][feature]
    else:
      model_cfg = data[model]
  elif model in ("DAROD", "T_RODNet"):
    with open(os.path.join('detection_vis_backend', 'networks', f'{model}_model_config.json'), 'r') as file:
      config = json.load(file)
    if dataset in config.keys():
      model_cfg = {k: v for k, v in config[dataset].items() if not k.startswith('_')}
  elif model in ("RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2", "RadarFormer_hrformer2d"):
    with open(os.path.join('detection_vis_backend', 'networks', 'RODNet_model_config.json'), 'r') as file:
      config = json.load(file)
    model_cfg = config[model]
  else:
    with open(os.path.join('detection_vis_backend', 'networks', f'{model}_model_config.json'), 'r') as file:
      data = json.load(file)
      model_cfg = data
  
  # train config
  if model in ("RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2", "RadarFormer_hrformer2d"):
    with open(os.path.join('detection_vis_backend', 'train', 'RODNet_train_config.json'), 'r') as file:
      data = json.load(file)
    train_cfg = data
  else:
    with open(os.path.join('detection_vis_backend', 'train', f'{model}_train_config.json'), 'r') as file:
      data = json.load(file)
    if 'default' in data.keys():
      train_cfg = data['default'].copy()  
      if dataset in data.keys():
        if feature[0] in data[dataset].keys():
          train_cfg.update(data[dataset][feature[0]]) 
        else:
          train_cfg.update(data[dataset]) 
    else:
      train_cfg = data
  return model_cfg, train_cfg


# Optinon 1: Train from scratch
# model_zoo = ["Choose a model type", "FFTRadNet", "RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2", "RECORD", "RECORDNoLstm", "RECORDNoLstmMulti", "MVRECORD", "RADDet", "DAROD", "RAMP_CNN", "RadarCrossAttention"]   # This will be changed when streamlit support selectbox with None as default option
model_zoo = ["Choose a model type"] + list(MODEL_FEATURES.keys())   # derived from model_features.json (single source of truth)

if train_mode == train_modes[0]:
  model = st.selectbox("Choose a model type:", model_zoo, index=0)
  if model != model_zoo[0]:
    # Feature-compatibility guard: every chosen input feature must be one this
    # model can consume (see networks/model_features.json). Sound subset check —
    # never blocks a valid combo; catches feeding a model a feature it can't read.
    # supported = MODEL_FEATURES.get(model, [])   # old flat schema: value was the feature list
    supported = MODEL_FEATURES.get(model, {}).get("features", [])   # value is now a dict; features live under "features"
    incompatible = [f for f in (st.session_state.features_chosen or []) if f not in supported]
    if incompatible:
      st.info(
        f"**{model}** can't train on the selected feature(s): {', '.join(incompatible)}. "
        f"It supports: {', '.join(supported)}. "
        "Adjust your feature selection on the Get features page, or pick another model."
      )
      st.stop()

    # Dataset-compatibility guard: the model must actually support the dataset the
    # chosen files come from (see networks/model_features.json "datasets"). This is
    # stronger than the feature check — e.g. FFTRadNet needs RADIal's specific MIMO
    # geometry, so an RD map from another dataset won't work even if it "is RD".
    # Skipped when a model has no "datasets" list yet (not-yet-confirmed models are
    # left un-gated rather than wrongly blocked).
    compatible_datasets = MODEL_FEATURES.get(model, {}).get("datasets", [])
    if compatible_datasets and dataset_name not in compatible_datasets:
      st.info(
        f"**{model}** can't train on the **{dataset_name}** dataset. "
        f"It supports: {', '.join(compatible_datasets)}. "
        "Choose files from a supported dataset, or pick another model."
      )
      st.stop()

    st.write("Model description")

    model_configs, train_configs = find_default_configs(dataset_name, st.session_state.features_chosen, model)

    config_editor(model_configs, train_configs)

    st.session_state.model_configs["class"] = model
    
    if st.session_state.split_mode == 'sequence':
      st.session_state.train_configs["dataloader"]["splitmode"] = "sequence"
      st.session_state.train_configs["dataloader"]["split_sequence"] = st.session_state.datafiles_split
    elif st.session_state.split_mode == 'random':
      st.session_state.train_configs["dataloader"]["splitmode"] = "random"
      st.session_state.train_configs["dataloader"]["split_random"] = st.session_state.split_ratios
    else:
      st.session_state.train_configs["dataloader"]["splitmode"] = "original"
    
    train_action = st.button("Train", key="train_btn")
    if train_action:
      if not check_sequence_split():
        st.stop()
      # Parsing of the chosen datafiles now happens inside the training job
      # (jobs/tasks.py:_parse_datafiles), so submit is a single instant enqueue.
      params = {"datafiles_chosen": st.session_state.datafiles_chosen,
                "features_chosen": st.session_state.features_chosen,
                "mlmodel_configs": st.session_state.model_configs,
                "train_configs": st.session_state.train_configs}
      response = requests.post(f"http://{backend_service}:8001/train", json=params)
      if response.status_code != 200:
        st.info("An error occurred in submitting training.")
        st.stop()
      st.session_state.train_job_id = response.json()["job_id"]
      st.success("Training job submitted. Track its progress on the Jobs page.")

    go_to_jobs_button("view_jobs_train")
      

        

def check_before_train(new_file_list, old_file_list):
  new_file_ids = [f['id'] for f in new_file_list]
  old_file_ids = [f['id'] for f in old_file_list]
  return not bool(set(new_file_ids) & set(old_file_ids))


# Option 2: Train on pre-trained model
if train_mode == train_modes[1]:
  # Get models list
  response = requests.get(f"http://{backend_service}:8001/models")  
  models = response.json()

  model_chosen = st.selectbox("Choose a model:", [item["name"] for item in models], index=0)
  model_id = [item['id'] for item in models if item['name'] == model_chosen][0]

  # Get model lineage info
  response = requests.get(f"http://{backend_service}:8001/model/{model_id}")  
  model_paras = response.json()

  config_editor(model_paras["model_config"], model_paras["train_config"])
  if st.session_state.split_mode == 'sequence':
    st.session_state.train_configs["dataloader"]["splitmode"] = "sequence"
    st.session_state.train_configs["dataloader"]["split_sequence"] = st.session_state.datafiles_split
  elif st.session_state.split_mode == 'random':
    st.session_state.train_configs["dataloader"]["splitmode"] = "random"
    st.session_state.train_configs["dataloader"]["split_random"] = st.session_state.split_ratios
  else:
    st.session_state.train_configs["dataloader"]["splitmode"] = "original"

  train_action = st.button("Train", key="train_btn")
  if train_action:
    if not check_sequence_split():
      st.stop()
    if not check_before_train(st.session_state.datafiles_chosen, model_paras["datafiles"]):
      st.info("The chosen data file(s) has been trained in this model, please change.")
    else:
      params = {"datafiles_chosen": st.session_state.datafiles_chosen,
                "features_chosen": st.session_state.features_chosen,
                "mlmodel_configs": st.session_state.model_configs,
                "train_configs": st.session_state.train_configs}
      response = requests.post(f"http://{backend_service}:8001/retrain/{model_id}", json=params)
      if response.status_code != 200:
        st.info("An error occurred in submitting retraining.")
        st.stop()
      st.session_state.retrain_job_id = response.json()["job_id"]
      st.success("Retraining job submitted. Track its progress on the Jobs page.")

  go_to_jobs_button("view_jobs_retrain")

infer_action = st.button("Go to inference", key="inference_btn")
if infer_action:
  switch_page("inference")

