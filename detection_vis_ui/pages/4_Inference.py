
import streamlit as st
import requests
import os
import re
import numpy as np 

from PIL import Image



backend_service = os.getenv('BACKEND_SERVICE', 'localhost')
model_rootdir = os.getenv('MODEL_ROOTDIR')

st.set_page_config(
    page_title="Have fun ",
    page_icon=":red_car:", 
)



response = requests.get(f"http://{backend_service}:8001/models")  
models = response.json()

if 'model_chosen' in st.session_state:
  model_index = next((idx for idx, item in enumerate(models) if item['name'] == st.session_state.model_chosen), -1)
else:
  model_index = 0
model_chosen = st.selectbox("Choose a model", [item["name"] for item in models], index=model_index)
model_id = [item['id'] for item in models if item['name'] == model_chosen][0]

st.subheader(f"{model_chosen} model")
st.write(model_id) # for debug

model_path = os.path.join(model_rootdir, model_chosen)

# in default choose the last epoch checkpoint model
files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
checkpoint_count = len(files) - 4
st.write(checkpoint_count) # for debug
checkpoint_id = st.selectbox("Choose a checkpoint(In default from the last epoch)", list(range(1,checkpoint_count+1)), index=checkpoint_count-1) - 1
st.write(checkpoint_id)

# Get data&model&train info of the chosen model 
response = requests.get(f"http://{backend_service}:8001/model/{model_id}")  
model_paras = response.json()
# st.json(model_paras["datafiles"])
# st.json(model_paras["features"])
# st.json(model_paras["model_config"])
# st.json(model_paras["train_config"])

# Get eval performance info of the chosen model


# Get sample split info of the chosen model
with open(os.path.join(model_path, "samples_split.txt"), 'r') as f:
  lines = f.readlines()
sample_ids = {}
for line in lines:
  if "TRAIN_SAMPLE_IDS:" in line:
    sample_ids["Train data"] = list(map(int, line.replace("TRAIN_SAMPLE_IDS: ", "").strip().split(',')))
  elif "VAL_SAMPLE_IDS:" in line:
    sample_ids["Val data"] = list(map(int, line.replace("VAL_SAMPLE_IDS: ", "").strip().split(',')))
  elif "TEST_SAMPLE_IDS:" in line:
    sample_ids["Test data"] = list(map(int, line.replace("TEST_SAMPLE_IDS: ", "").strip().split(',')))  


radio_options = ["Train data", "Val data", "Test data", "I want to upload my own data file to do inference"]
radio_select = st.radio("Which data you want to check the detection model from?", radio_options)

if radio_select is radio_options[-1]:
  uploaded_file = st.file_uploader("Choose a data file", accept_multiple_files=False)
  if uploaded_file is not None:
    #bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    im = Image.open(uploaded_file)
    im = im.resize((600, 400))
    image = np.array(im)
    st.image(image)
else:
  frame_begin = 0
  frame_end = len(sample_ids[radio_select])
  frame_id = st.slider('Choose a data frame', frame_begin, frame_end, frame_begin)
  st.write(f"No. {sample_ids[radio_select][frame_id]} sample")
  params = {"checkpoint_id": checkpoint_id, "sample_id": sample_ids[radio_select][frame_id]}
  response = requests.get(f"http://{backend_service}:8001/predict/{model_id}", params=params)  
  res = response.json()
  pred_image = np.array(res["prediction"])
  st.image(pred_image, caption="Ground Truth & predicted objects")
