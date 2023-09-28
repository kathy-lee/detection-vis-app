
import streamlit as st
import requests
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from streamlit_extras.switch_page_button import switch_page
from streamlit_ace import st_ace



backend_service = os.getenv('BACKEND_SERVICE', 'localhost')
model_rootdir = os.getenv('MODEL_ROOTDIR')

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
  s += "- " + i["name"] + "\n"
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
    if dataset_name in ("RADIal", "CARRADA", "RADDet_dataset"):
      split_mode = st.radio("Select the data split way:", ["sequence", "random", "original"], horizontal=True, key=f"radio_splitmode")
      st.write("Original mode is the default split which the dataset provides.")
    else:
      split_mode = st.radio("Select the data split way:", ["sequence", "random"], horizontal=True, key=f"radio_splitmode")
  else:
    st.info("Train with files from different datasets is not supported currently.")
    st.stop()
else:
  if dataset_name in ("RADIal", "CARRADA", "RADDet_dataset"):
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
  if 'datafiles_split' not in st.session_state:
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
  model_configs["class"] = model
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


# Optinon 1: Train from scratch
model_zoo = ["Choose a model type", "FFTRadNet", "RODNet", "RECORD"]   # This will be changed when streamlit support selectbox with None as default option

if train_mode == train_modes[0]:
  model = st.selectbox("Choose a model type:", model_zoo, index=0)
  if model != model_zoo[0]:  
    st.write("Model description")

    with open(os.path.join('detection_vis_backend', 'networks', 'default_model_config.json'), 'r') as file:
      data = json.load(file)
      model_configs = data[model]
    
    with open(os.path.join('detection_vis_backend', 'train', f'{model}_train_config.json'), 'r') as file:
      data = json.load(file)
      train_configs = data

    config_editor(model_configs, train_configs)
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
      with st.spinner(text="Training model in progress..."):
        # find the chosen data files which are not parsed yet, and parse them
        for f in st.session_state.datafiles_chosen:
          datafile_parsed = f"parse_{f['id']}"
          if datafile_parsed not in st.session_state:
            st.session_state[datafile_parsed] = False
            response = requests.get(f"http://{backend_service}:8001/parse/{f['id']}")
            if response.status_code != 204:
              st.info(f"An error occurred in parsing data file {f['name']}.")
              st.stop()
            else:
              st.session_state[datafile_parsed] = True
        params = {"datafiles_chosen": st.session_state.datafiles_chosen, 
                  "features_chosen": st.session_state.features_chosen, 
                  "mlmodel_configs": st.session_state.model_configs, 
                  "train_configs": st.session_state.train_configs,
                  "use_original_split": st.session_state.split_mode == "original"}
        response = requests.post(f"http://{backend_service}:8001/train", json=params)
      if response.status_code != 200:
        st.info("An error occurred in training.")
        st.stop()
      else:
        res = response.json()
        st.session_state.model_chosen = res["model_name"]
        # show evaluation result
        st.write("Model Evaluation")
        st.write("Val dataset")
        val_eval = pd.read_csv(os.path.join(model_rootdir, st.session_state.model_chosen, "val_eval.csv"))
        st.table(val_eval)
        st.write("Test dataset")
        test_eval = pd.read_csv(os.path.join(model_rootdir, st.session_state.model_chosen, "test_eval.csv"))
        st.table(test_eval)         
      

        

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
    if not check_before_train(st.session_state.datafiles_chosen, model_paras["datafiles"]):
      st.info("The chosen data file(s) has been trained in this model, please change.")
    else:
      with st.spinner(text="Training model in progress..."):
        params = {"datafiles_chosen": st.session_state.datafiles_chosen, 
                  "features_chosen": st.session_state.features_chosen, 
                  "mlmodel_configs": st.session_state.model_configs, 
                  "train_configs": st.session_state.train_configs,
                  "use_original_split": st.session_state.split_mode == "original"}
        response = requests.post(f"http://{backend_service}:8001/retrain/{model_id}", json=params)
      if response.status_code != 200:
        st.info("An error occurred in training.")
        st.stop()
      else:
        res = response.json()
        st.session_state.model_chosen = res["model_name"]
        # show evaluation result
        st.write("Model Evaluation")
        st.write("Val dataset")
        val_eval = pd.read_csv(os.path.join(model_rootdir, st.session_state.model_chosen, "val_eval.csv"))
        st.table(val_eval)
        st.write("Test dataset")
        test_eval = pd.read_csv(os.path.join(model_rootdir, st.session_state.model_chosen, "test_eval.csv"))
        st.table(test_eval)

infer_action = st.button("Go to inference", key="inference_btn")
if infer_action:
  switch_page("inference")

