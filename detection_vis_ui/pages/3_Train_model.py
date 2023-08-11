
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


if 'datafiles_chosen' not in st.session_state or 'features_chosen' not in st.session_state:
  st.info("Please choose data and feature first.")
  st.stop()

datafiles_chosen = st.session_state.datafiles_chosen
features_chosen = st.session_state.features_chosen

st.write("You have chosen the following data file(s):")
s = ''
for i in st.session_state.datafiles_chosen:
  s += "- " + i["name"] + "\n"
st.markdown(s)
edit_data_action = st.button("Edit data", key="edit_data_btn")
if edit_data_action:
  switch_page("choose data")


features_chosen = st.write("You have chosen the following feature(s):")
s = ''
for i in st.session_state.features_chosen:
  s += "- " + i + "\n"
st.markdown(s)
edit_feature_action = st.button("Edit feature", key="edit_feature_btn")
if edit_feature_action:
  switch_page("get features")

train_modes = ["Train from scratch", "Train on a pre-trained model"]
train_mode = st.radio("Choose a way to train model:", train_modes)


# Optinon 1: Train from scratch
model_zoo = ["Choose a model type", "FFTRadNet", "ABCNet"]   # This will be changed when streamlit support selectbox with None as default option
# classif_loss_functions = ["FocalLoss"]
# reg_loss_functions = ["smoothL1Loss"]

if train_mode == train_modes[0]:
  model = st.selectbox("Choose a model type:", model_zoo, index=0)
  if model != model_zoo[0]:  
    if "model_configs" not in st.session_state:
      st.session_state.model_configs = {}
    if "train_configs" not in st.session_state:
      st.session_state.train_configs = {}
    if "model_chosen" not in st.session_state:
      st.session_state.model_chosen = None

    st.write("Model description")

    with open(os.path.join('detection_vis_backend', 'networks', 'default_model_config.json'), 'r') as file:
      data = json.load(file)
      model_configs = data["fftradnet"]
    
    with open(os.path.join('detection_vis_backend', 'train', 'default_train_config.json'), 'r') as file:
      data = json.load(file)
      train_configs = data

    st.write("Model config:")
    formatted_model_configs = json.dumps(model_configs, indent=2)
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
    model_configs["type"] = model
    st.session_state.model_configs = model_configs
    
    st.write("Train config:")
    formatted_train_configs = json.dumps(train_configs, indent=2)
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

    # modelconfig_expander = st.expander("Model Config", expanded=True)
    # trainconfig_expander = st.expander("Train Config", expanded=True)
    
    # with modelconfig_expander:
    #   if model == "FFTRadNet":
    #     blocks = [None] * 4
    #     blocks[0] = st.number_input("backbone_block_1", min_value=1, max_value=10, value=3)
    #     blocks[1] = st.number_input("backbone_block_2", min_value=1, max_value=10, value=6)
    #     blocks[2] = st.number_input("backbone_block_3", min_value=1, max_value=10, value=6)
    #     blocks[3] = st.number_input("backbone_block_4", min_value=1, max_value=10, value=3)
    #     channels = [None] * 4
    #     channels[0] = st.number_input("backbone_channel_1", min_value=1, max_value=100, value=32)
    #     channels[1] = st.number_input("backbone_channel_2", min_value=1, max_value=100, value=40)
    #     channels[2] = st.number_input("backbone_channel_3", min_value=1, max_value=100, value=48)
    #     channels[3] = st.number_input("backbone_channel_4", min_value=1, max_value=100, value=56)
    #     mimolayer = st.number_input("MIMO layer output", min_value=64, max_value=256, value=192)
    #     detectionhead = st.radio("Detection Head", ["True", "False"])
    #     seghead = st.radio("Segmentation Head", ["True", "False"])
    #     st.session_state.model_configs["type"] = model
    #     st.session_state.model_configs["blocks"] = blocks
    #     st.session_state.model_configs["channels"] = channels
    #     st.session_state.model_configs["mimo_layer"] = mimolayer
    #     st.session_state.model_configs["detection_head"] = detectionhead
    #     st.session_state.model_configs["segmentation_head"] = seghead


    # with trainconfig_expander:
    #   seed = st.number_input("Random seed:", min_value=1, max_value=10, value=3)
    #   epoch = st.number_input("Num of epochs:", min_value=1, max_value=100, value=1)
    #   lr = st.number_input("Initial learning rate for the optimizer:", min_value=1e-5, max_value=1e-2, value=1e-4, format="%.5f")
    #   step_size = st.number_input("Step size of learning rate scheduling:", min_value=1, max_value=20, value=10)
    #   gamma = st.number_input("Gamma factor of learning rate scheduling:", min_value=0.1, max_value=1.0, value=0.9, format="%.5f")
    #   classif_loss = st.selectbox("Loss function of classification:", classif_loss_functions, index=0)
    #   reg_loss = st.selectbox("Loss function of regression:", reg_loss_functions, index=0)
    #   st.session_state.train_configs["seed"] = seed
    #   st.session_state.train_configs["num_epochs"] = epoch
    #   st.session_state.train_configs["lr"] = lr
    #   st.session_state.train_configs["step_size"] = step_size
    #   st.session_state.train_configs["gamma"] = gamma
    #   st.session_state.train_configs["losses"] = {"classification": classif_loss, "regression": reg_loss, "weight": [1,100,100]}
    #   st.session_state.train_configs["mode"] = "sequence"
    #   st.session_state.train_configs["train"] = {"batch_size": 4, "num_workers": 4}
    #   st.session_state.train_configs["val"] = {"batch_size": 4, "num_workers": 4}
    #   st.session_state.train_configs["test"] = {"batch_size": 1, "num_workers": 1}
    
    train_action = st.button("Train", key="train_btn")
    if train_action:
      with st.spinner(text="Training model in progress..."):
        params = {"datafiles_chosen": st.session_state.datafiles_chosen, 
                  "features_chosen": st.session_state.features_chosen, 
                  "mlmodel_configs": st.session_state.model_configs, 
                  "train_configs": st.session_state.train_configs}
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
        infer_action = st.button("Go to inference", key="inference_btn")
        if infer_action:
          switch_page("inference")
        

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

  # show model lineage(datafiles, features, model config, train config, performance)
  # In retrain case, features and model type could not be changed, datafiles will check if repeated
  # st.data_editor(model_paras["model_config"], disabled=("type"), column_config = {"blocks": None, "channels": None})
  # st.data_editor(model_paras["train_config"])
  formatted_model_config = json.dumps(model_paras["model_config"], indent=2)
  new_model_config = st_ace(value=formatted_model_config,theme="solarized_dark",
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
  formatted_train_config = json.dumps(model_paras["train_config"], indent=2)
  new_train_config = st_ace(value=formatted_train_config,theme="solarized_dark",
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
  train_action = st.button("Train", key="train_btn")
  if train_action:
    if not check_before_train(datafiles_chosen, model_paras["datafiles"]):
      st.info("The chosen data file(s) has been trained in this model, please change.")
    else:
      with st.spinner(text="Training model in progress..."):
        params = {"datafiles_chosen": datafiles_chosen, 
                  "features_chosen": features_chosen, 
                  "mlmodel_configs": new_model_config, 
                  "train_configs": new_train_config}
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

