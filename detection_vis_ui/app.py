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

backend_service = os.getenv('BACKEND_SERVICE', 'localhost')

# ########################
# response = requests.get("http://detection_vis_backend:8001/users")  # or your FastAPI server URL
# users = response.json()
# st.json(users)
# ########################

response = requests.get(f"http://{backend_service}:8001/datasets")  # or your FastAPI server URL
datasets = response.json()

# show datasets on top level
root_datasets = []
for d in datasets:
  if d["parent_id"] is None:
    root_datasets.append(d)
dataset_name = st.selectbox("Choose the dataset", [i["name"] for i in root_datasets])
st.subheader(f"{dataset_name} Dataset")

# get the chosen dataset
dataset = {}
for d in datasets:
  if d["name"] == dataset_name:
    dataset = d
    st.write(d["description"])

# get subdatasets from the chosen dataset
subdatasets = []
for d in datasets:
  if d["parent_id"] == dataset["id"]:
    subdatasets.append(d)

if 'datafiles_chosen' not in st.session_state:
  st.session_state.datafiles_chosen = []
if 'checkbox' not in st.session_state:
  st.session_state.checkbox = []

if subdatasets:
  # show subdatasets in tab pages
  subdataset_tabs = st.tabs([i["name"] for i in subdatasets])
  index = 0
  for i, t in enumerate(subdataset_tabs):
    with t:
      # Show data files 
      response = requests.get(f"http://{backend_service}:8001/dataset/{subdatasets[index]['id']}")
      files = response.json()

      colms = st.columns((1, 2, 2, 1, 1))
      fields = ['No', 'File name', 'info', "size", 'select']
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
        if f"{dataset['name']}_{index}_{file['name']}" not in st.session_state:
          checkbox_status = col5.checkbox("", value=False,key=f"checkbox_{dataset['name']}_{index}_{file['name']}")
          st.session_state[f"{dataset['name']}_{index}_{file['name']}"] = False
        elif st.session_state[f"{dataset['name']}_{index}_{file['name']}"]:
          checkbox_status = col5.checkbox("", value=True,key=f"checkbox_{dataset['name']}_{index}_{file['name']}")
        else:
          checkbox_status = col5.checkbox("", value=False,key=f"checkbox_{dataset['name']}_{index}_{file['name']}")

        if checkbox_status and file not in st.session_state.datafiles_chosen:
          st.session_state.datafiles_chosen.append(file)
          st.session_state[f"{dataset['name']}_{index}_{file['name']}"] = True

        if not checkbox_status and file in st.session_state.datafiles_chosen:
          st.session_state[f"{dataset['name']}_{index}_{file['name']}"] = False
          st.session_state.datafiles_chosen.remove(file)

    index = index + 1
else:
  # Show data files 
  response = requests.get(f"http://{backend_service}:8001/dataset/{dataset['id']}")
  files = response.json()

  colms = st.columns((1, 2, 2, 1, 1))
  fields = ['No', 'File name', 'info', "size", 'select']
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
    if f"{dataset['name']}_{file['name']}" not in st.session_state:
      checkbox_status = col5.checkbox("", value=False,key=f"checkbox_{dataset['name']}_{file['name']}")
      st.session_state[f"{dataset['name']}_{file['name']}"] = False
    elif st.session_state[f"{dataset['name']}_{file['name']}"]:
      checkbox_status = col5.checkbox("", value=True,key=f"checkbox_{dataset['name']}_{file['name']}")
    else:
      checkbox_status = col5.checkbox("", value=False,key=f"checkbox_{dataset['name']}_{file['name']}")

    if checkbox_status and file not in st.session_state.datafiles_chosen:
      st.session_state.datafiles_chosen.append(file)
      st.session_state[f"{dataset['name']}_{file['name']}"] = True

    if not checkbox_status and file in st.session_state.datafiles_chosen:
      st.session_state[f"{dataset['name']}_{file['name']}"] = False
      st.session_state.datafiles_chosen.remove(file)
    

st.write(st.session_state.datafiles_chosen)
button_click = st.button("Check feature")
if button_click:
  #check_datafiles(st.session_state.datafiles_chosen)
  switch_page("get features")
