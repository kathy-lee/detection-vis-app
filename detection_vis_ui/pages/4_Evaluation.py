
import streamlit as st
import requests
import os
import numpy as np 

from PIL import Image



backend_service = os.getenv('BACKEND_SERVICE', 'localhost')

st.set_page_config(
    page_title="Have fun ",
    page_icon=":red_car:", 
)

subdatasets = ["Train data", "Val data", "Test data", "I want to upload my own data file to do inference"]
radio_option = st.radio("Which frame you want to check the detection model from?", subdatasets)

if radio_option is not subdatasets[-1]:
  frame_begin = 0
  frame_end = 10
  frame_id = st.slider('Choose a data frame', frame_begin, frame_end, frame_begin)
else:
  uploaded_file = st.file_uploader("Choose a data file", accept_multiple_files=False)
  if uploaded_file is not None:
    #bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    im = Image.open(uploaded_file)
    image = np.array(im)
    st.image(image)


#response = requests.get(f"http://{backend_service}:8001/inference/{model}")

st.write("Ground Truth")
st.write("Detected")