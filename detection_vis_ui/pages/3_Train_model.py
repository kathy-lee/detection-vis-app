
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
    layout = "wide"
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
edit_data_action = st.button("Edit", key="edit_data_btn")
if edit_data_action:
  switch_page("choose data")


features_chosen = st.write("You have chosen the following feature(s):")
s = ''
for i in st.session_state.features_chosen:
  s += "- " + i + "\n"
st.markdown(s)
edit_feature_action = st.button("Edit", key="edit_feature_btn")
if edit_feature_action:
  switch_page("get features")

train_modes = ["Yes", "No, I want to train an existed model", "I want to do inference on an exisited model"]
train_mode = st.radio("Would you like to train a model from scratch?", train_modes)

model_zoo = ["Choose a model", "FFTRadNet", "ABCNet"]   # This will be changed when streamlit support selectbox with None as default option
if train_mode == train_modes[0]:
  model = st.selectbox("Choose a model:", model_zoo, index=0)
  if model != model_zoo[0]:  
    st.write("Model description")
    modelconfig_expander = st.expander("Model Config")
    with modelconfig_expander:
      st.write("model para")
      
    trainconfig_expander = st.expander("Train Config")
    with trainconfig_expander:
      st.write("train para")
    
    train_action = st.button("Train", key="train_btn")
    evaluation_action = st.button("Evaluation", key="evaluation_btn")
    if train_action:
      with st.spinner(text="Training model in progress..."):
        response = requests.get(f"http://{backend_service}:8001/train/{model}")
      if response.status_code != 204:
        st.info("An error occurred in training.")
        st.stop()
    if evaluation_action:
      switch_page("evaluation")
