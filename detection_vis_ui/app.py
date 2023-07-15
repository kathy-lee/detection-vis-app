import requests
import os
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd

st.set_page_config(
    page_title="Have fun ",
    page_icon=":red_car:", 
)

# Define the title 
st.title("Automotive sensor data visualization web application")

# ########################
# response = requests.get("http://detection_vis_backend:8001/users")  # or your FastAPI server URL
# users = response.json()
# st.json(users)
# ########################

response = requests.get("http://localhost:8001/datasets")  # or your FastAPI server URL
datasets = response.json()

# show datasets on top level
root_datasets = []
for d in datasets:
  if d["parent_id"] is None:
    root_datasets.append(d)
dataset_name = st.selectbox("Choose the dataset", [i["name"] for i in root_datasets])
st.subheader(f"{dataset_name} Dataset")

dataset = {}
for d in datasets:
  if d["name"] == dataset_name:
    dataset = d
    st.write(d["description"])

subdatasets = []
for d in datasets:
  if d["parent_id"] == dataset["id"]:
    subdatasets.append(d)

  datafile_chosen = {}
if subdatasets:
  # show subdataset in tab page
  subdataset_tabs = st.tabs([i["name"] for i in subdatasets])
  index = 0
  button_index = 0
  for t in subdataset_tabs:
    with t:
      # Show data files 
      response = requests.get(f"http://localhost:8001/dataset/{subdatasets[index]['id']}")
      files = response.json()

      colms = st.columns((1, 2, 2, 1, 1))
      fields = ['No', 'File name', 'info', "size", 'action']
      # table header
      for col, field_name in zip(colms, fields):
        col.write(field_name)
      # table content
      for no, file in enumerate(files):
        col1, col2, col3, col4, col5 = st.columns((1, 2, 2, 1, 1))
        col1.write(no)  
        col2.write(file["name"])  
        col3.write(file["description"])  
        col4.write(file["size"]) 
        # last column
        button_type = "Load" 
        button_phold = col5.empty()  # create a placeholder
        do_action = button_phold.button(button_type, key=button_index)
        button_index = button_index + 1
        if do_action:
          datafile_chosen = file
          if 'datafile_chosen' not in st.session_state:
            st.session_state['datafile_chosen'] = datafile_chosen
          button_phold.empty()  #  remove button
          loading_phold = col5.empty()  # create another placeholder for loading
          loading_phold.write('Loading...')
          params = {
            "file_path": file["path"],
            "file_name": file["name"],
          }
          response = requests.get("http://localhost:8001/download", params=params)
          if response.status_code == 204:
            loading_phold.empty()  # remove loading text
            col5.write('Loaded')
            switch_page("get features")
    index = index + 1
else:
  button_index = 0
  # Show data files 
  response = requests.get(f"http://localhost:8001/dataset/{dataset['id']}")
  files = response.json()

  colms = st.columns((1, 2, 2, 1, 1))
  fields = ['No', 'File name', 'info', "size", 'action']
  # table header
  for col, field_name in zip(colms, fields):
    col.write(field_name)
  # table content
  for no, file in enumerate(files):
    col1, col2, col3, col4, col5 = st.columns((1, 2, 2, 1, 1))
    col1.write(no)  
    col2.write(file["name"])  
    col3.write(file["description"])  
    col4.write("1GB") 
    # last column
    button_type = "Load" 
    button_phold = col5.empty()  # create a placeholder
    do_action = button_phold.button(button_type, key=button_index)
    button_index = button_index + 1
    if do_action:
      datafile_chosen = file
      if 'datafile_chosen' not in st.session_state:
        st.session_state['datafile_chosen'] = datafile_chosen
      button_phold.empty()  #  remove button
      loading_phold = col5.empty()  # create another placeholder for loading
      loading_phold.write('Loading...')
      params = {
        "file_path": file["path"],
        "file_name": file["name"],
      }
      response = requests.get("http://localhost:8001/download", params=params)
      if response.status_code == 204:
        loading_phold.empty()  # remove loading text
        col5.write('Loaded')
        switch_page("get features")


