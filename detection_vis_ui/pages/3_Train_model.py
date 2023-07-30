
import streamlit as st
import requests
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from streamlit_extras.switch_page_button import switch_page



backend_service = os.getenv('BACKEND_SERVICE', 'localhost')

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

train_modes = ["Yes", "No, I want to train on a pre-trained model", "No, I want to choose a model and do inference"]
train_mode = st.radio("Would you like to train a model from scratch?", train_modes)


# Optinon 1: Train from scratch
model_zoo = ["Choose a model", "FFTRadNet", "ABCNet"]   # This will be changed when streamlit support selectbox with None as default option
classif_loss_functions = ["FocalLoss"]
reg_loss_functions = ["smoothL1Loss"]

if train_mode == train_modes[0]:
  model = st.selectbox("Choose a model:", model_zoo, index=0)
  if model != model_zoo[0]:  
    if "model_configs" not in st.session_state:
      st.session_state.model_configs = {}
    if "train_configs" not in st.session_state:
      st.session_state.train_configs = {}

    st.write("Model description")
    modelconfig_expander = st.expander("Model Config")
    trainconfig_expander = st.expander("Train Config")
    
    with modelconfig_expander:
      if model == "FFTRadNet":
        blocks = [None] * 4
        blocks[0] = st.number_input("backbone_block_1", min_value=1, max_value=10, value=3)
        blocks[1] = st.number_input("backbone_block_2", min_value=1, max_value=10, value=6)
        blocks[2] = st.number_input("backbone_block_3", min_value=1, max_value=10, value=6)
        blocks[3] = st.number_input("backbone_block_4", min_value=1, max_value=10, value=3)
        channels = [None] * 4
        channels[0] = st.number_input("backbone_channel_1", min_value=1, max_value=100, value=32)
        channels[1] = st.number_input("backbone_channel_2", min_value=1, max_value=100, value=40)
        channels[2] = st.number_input("backbone_channel_3", min_value=1, max_value=100, value=48)
        channels[3] = st.number_input("backbone_channel_4", min_value=1, max_value=100, value=56)
        mimolayer = st.number_input("MIMO layer output", min_value=64, max_value=256, value=192)
        detectionhead = st.radio("Detection Head", ["True", "False"])
        seghead = st.radio("Segmentation Head", ["True", "False"])
        st.session_state.model_configs["blocks"] = blocks
        st.session_state.model_configs["channels"] = channels
        st.session_state.model_configs["mimolayer"] = mimolayer
        st.session_state.model_configs["detectionhead"] = detectionhead
        st.session_state.model_configs["seghead"] = seghead


    with trainconfig_expander:
      epoch = st.number_input("Num of epochs:", min_value=1, max_value=100, value=100)
      lr = st.number_input("Initial learning rate for the optimizer:", min_value=1e-5, max_value=1e-2, value=1e-4, format="%.5f")
      step_size = st.number_input("Step size of learning rate scheduling:", min_value=1, max_value=20, value=10)
      gamma = st.number_input("Gamma factor of learning rate scheduling:", min_value=0.1, max_value=1.0, value=0.9, format="%.5f")
      classif_loss = st.selectbox("Loss function of classification:", classif_loss_functions, index=0)
      reg_loss = st.selectbox("Loss function of regression:", reg_loss_functions, index=0)
      st.session_state.train_configs["epoch"] = epoch
      st.session_state.train_configs["lr"] = lr
      st.session_state.train_configs["step_size"] = step_size
      st.session_state.train_configs["gamma"] = gamma
      st.session_state.train_configs["classif_loss"] = classif_loss
      st.session_state.train_configs["reg_loss"] = reg_loss
      st.session_state.train_configs["weight"] = [1,100,100]
    
    train_action = st.button("Train", key="train_btn")
    evaluation_action = st.button("Evaluation", key="evaluation_btn")
    if train_action:
      with st.spinner(text="Training model in progress..."):
        params = {"datafiles_chosen": st.session_state.datafiles_chosen, 
                  "features_chosen": st.session_state.features_chosen, 
                  "mlmodel_configs": st.session_state.model_configs, 
                  "train_configs": st.session_state.train_configs}
        response = requests.post(f"http://{backend_service}:8001/train", json=params)
      if response.status_code != 204:
        st.info("An error occurred in training.")
        st.stop()
    if evaluation_action:
      switch_page("evaluation")

# Option 2: Train on pre-trained model


# Option 3: Do inference
if train_mode == train_modes[2]:
  # show data files
  # show features
  # show model architecture and config
  # show training config
  # show evaluation
  model = st.selectbox("Choose a model:", model_zoo, index=0)