
import streamlit as st
import numpy as np
import pandas as pd
import requests
import os

from streamlit_extras.switch_page_button import switch_page



st.set_page_config(
    page_title="Have fun ",
    page_icon=":red_car:", 
)


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
dataset_name = st.selectbox("Choose a dataset", [i["name"] for i in root_datasets])
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
        checkbox = f"{dataset['name']}_{index}_{file['name']}"
        if  checkbox not in st.session_state:
          checkbox_status = col5.checkbox(" ", value=False,key=f"checkbox_{checkbox}")
          st.session_state[checkbox] = False
        elif st.session_state[checkbox]:
          checkbox_status = col5.checkbox(" ", value=True,key=f"checkbox_{checkbox}")
        else:
          checkbox_status = col5.checkbox(" ", value=False,key=f"checkbox_{checkbox}")

        if checkbox_status and file not in st.session_state.datafiles_chosen:
          st.session_state.datafiles_chosen.append(file)
          st.session_state[checkbox] = True

        if not checkbox_status and file in st.session_state.datafiles_chosen:
          st.session_state[checkbox] = False
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
    checkbox = f"{dataset['name']}_{file['name']}"
    if checkbox not in st.session_state:
      checkbox_status = col5.checkbox(" ", value=False,key=f"checkbox_{checkbox}")
      st.session_state[f"{dataset['name']}_{file['name']}"] = False
    elif st.session_state[f"{dataset['name']}_{file['name']}"]:
      checkbox_status = col5.checkbox(" ", value=True,key=f"checkbox_{checkbox}")
    else:
      checkbox_status = col5.checkbox(" ", value=False,key=f"checkbox_{checkbox}")

    if checkbox_status and file not in st.session_state.datafiles_chosen:
      st.session_state.datafiles_chosen.append(file)
      st.session_state[f"{dataset['name']}_{file['name']}"] = True

    if not checkbox_status and file in st.session_state.datafiles_chosen:
      st.session_state[f"{dataset['name']}_{file['name']}"] = False
      st.session_state.datafiles_chosen.remove(file)
    

# st.write(st.session_state.datafiles_chosen)
button_click = st.button("Check feature")
if button_click:
  #check_datafiles(st.session_state.datafiles_chosen)
  switch_page("get features")


################################################
# import requests
# import os
# import time
# import numpy as np
# import matplotlib.pyplot as plt

# complex_array = np.random.rand(32, 8, 304) + 1j * np.random.rand(32, 8, 304)
# print(complex_array[0,0,0])
# serialized_feature = [[[(x.real, x.imag) for x in y] for y in z] for z in complex_array.tolist()]
# complex_feature = np.array([[[complex(real, imag) for real, imag in y] for y in z] for z in serialized_feature])
# print(complex_feature[0,0,0])
# abs_array = np.abs(complex_feature[:,:,0])
# print(abs_array.shape)

# import plotly.graph_objs as go

# x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
# y = x.copy().T # transpose
# z = np.cos(x ** 2 + y ** 2)
# st.info(x.shape)
# st.info(y.shape)
# st.info(z.shape)
# trace = go.Surface(x = x, y = y, z =z )
# data = [trace]
# layout = go.Layout(title = '3D Surface plot')
# fig = go.Figure(data = data)
# st.plotly_chart(fig, use_container_width=True)
##############################################################



########################################################################
# Don't change!
# def show_next(i, length):
#   # Increments the counter to get next photo
#   st.session_state[i] += 1
#   if st.session_state[i] >= length:
#     st.session_state[i] = 0
  

# def show_last(i, length):
#   # Decrements the counter to get next photo
#   st.session_state[i] -= 1
#   if st.session_state[i] < 0:
#     st.session_state[i] = length-1


# rawset = ["radar", "image", "lidar"]
# featureset = ["radarstatus", "imagestatus", "lidarstatus"]
# status = [True, True, False]

# # test expander with button 
# expanders = [None]*3
# # for idx,i in enumerate(rawset):
# for idx, (i, f) in enumerate(zip(rawset, featureset)):
#   expanders[idx] = st.expander(i)
#   with expanders[idx]:
#     if i not in st.session_state: 
#       st.session_state[i] = 0
#     if f not in st.session_state:
#       st.session_state[f] = status[idx]
#     if st.session_state[f]:
#       fileset = [os.path.join(i,f) for f in os.listdir(i)]
#       st.write(f"Total frames: {len(fileset)}")
#       forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([i,len(fileset)]), key=f"{i}_forward_btn")
#       backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([i,len(fileset)]), key=f"{i}_backward_btn")
#       photo = fileset[st.session_state[i]]
#       st.image(photo,caption=photo)
#       st.write(f"Index : {st.session_state[i]}")
#     else:
#       #placeholder[idx]=
#       placeholder1 = st.empty()
#       placeholder2 = st.empty()
#       placeholder3 = st.empty()
#       placeholder4 = st.empty()
#       placeholder5 = st.empty()
#       placeholder6 = st.empty()
      
#       load_action = placeholder1.button("Get feature")
#       # if "load_state" not in st.session_state:
#       #   st.session_state["load_state"] = False
#       if load_action:# or st.session_state["load_state"]:
#         #st.session_state["load_state"] = True
#         placeholder1.empty()
#         st.session_state['lidarstatus'] = True
#         i = 'lidar'
#         with expanders[2]:
#           fileset = [os.path.join(i,f) for f in os.listdir(i)]
#           placeholder1.empty()
#           placeholder2.write(f"Total frames: {len(fileset)}")
#           forward_btn = placeholder3.button("Show next frame ⏭️",on_click=show_next,args=([i,len(fileset)]), key=f"{i}_forward_btn")
#           backward_btn = placeholder4.button("Show last frame ⏪",on_click=show_last,args=([i,len(fileset)]), key=f"{i}_backward_btn")
#           photo = fileset[st.session_state[i]]
#           placeholder5.image(photo,caption=photo)
#           placeholder6.write(f"Index : {st.session_state[i]}")
            

#st.info(f"load state: {st.session_state['load_state']}")

########################################################################
# # layout of the page
# grid = make_grid(2,1)

# datafile_chosen = {}
# datafile_chosen["spectrogram"] = False
# datafile_chosen["image"] = True
# datafile_chosen["depth_image"] = False


# if datafile_chosen["spectrogram"]:
#   grid[0][0].header(f"Spectrogram feature")

#   col1_spectrogram,col2_spectrogram = grid[0][0].columns(2)

#   col1_spectrogram.write(f"Total frames: 1000")

#   fig, ax = plt.subplots(figsize=(5, 5))
#   ax.imshow(np.random.rand(8, 90), interpolation='nearest', aspect='auto')
#   col2_spectrogram.pyplot(fig, use_container_width=False)


# # show camera images
# if datafile_chosen["image"]:
#   grid[1][0].header(f"Camera images")

#   col1_image,col2_image = grid[1][0].columns(2)

#   photo = np.random.rand(8, 90)
#   fig, ax = plt.subplots(figsize=(5, 5))
#   ax.imshow(photo, aspect='auto')
#   col2_image.pyplot(fig, use_container_width=False)

#   col1_image.write(f"Index : ")

# if datafile_chosen["depth_image"]:
#   grid[2][0].header(f"Depth images")

#   col1_depth,col2_depth = grid[2][0].columns(2)

#   photo = np.random.rand(8, 90)
#   fig, ax = plt.subplots(figsize=(5, 5))
#   ax.imshow(photo, aspect='auto')
#   col2_depth.pyplot(fig, use_container_width=False)

#   col1_depth.write(f"Index : ")
