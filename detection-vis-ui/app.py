import requests
import paramiko
import os
import streamlit as st
import pandas as pd

# Define the title 
st.title("Automotive sensor data visualization web application")

# ########################
# response = requests.get("http://detection-vis-backend:8001/users")  # or your FastAPI server URL
# users = response.json()
# st.json(users)
# ########################

response = requests.get("http://detection-vis-backend:8001/datasets")  # or your FastAPI server URL
datasets = response.json()
# st.json(datasets)
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

if subdatasets:
  # show subdataset in tab page
  subdataset_tabs = st.tabs([i["name"] for i in subdatasets])
  index = 0
  button_index = 0
  for t in subdataset_tabs:
    with t:
      # Show data files 
      response = requests.get(f"http://detection-vis-backend:8001/dataset/{subdatasets[index]['id']}")
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
          bagfile = file
          # response = requests.post(f"http://backend:8080/load/{file}", file=bagfile)
          # datafile = response.json()
          button_phold.empty()  #  remove button
          col5.write('Loading...')
    index = index + 1
else:
  button_index = 0
  # Show data files 
  response = requests.get(f"http://detection-vis-backend:8001/dataset/{dataset['id']}")
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
      button_phold.empty()  #  remove button
      loading_phold = col5.empty()  # create another placeholder for loading
      loading_phold.write('Loading...')
      
      #
      #response = requests.post("http://localhost:8000/download", data=json.dumps({"file": selected_file}), headers={'Content-Type': 'application/json'})
      # if response.status_code == 200:
      #     st.write("File download initiated")
      # download data file from remote
      ssh = paramiko.SSHClient()
      ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically add the server's SSH key (not recommended for production)
      private_key = paramiko.RSAKey.from_private_key_file('key_for_ssh')
      ssh.connect('mifcom-desktop', username='kangle', pkey=private_key)
      sftp = ssh.open_sftp()
      #remote_file_path = os.path.join("/home/kangle", file["path"], file["name"])
      remote_file_path = os.path.join("/home/kangle", file["path"], "test.txt")
      local_file_path = file["name"]
      sftp.get(remote_file_path, local_file_path)
      sftp.close()
      ssh.close()
      loading_phold.empty()  # remove loading text
      col5.write('Loaded')