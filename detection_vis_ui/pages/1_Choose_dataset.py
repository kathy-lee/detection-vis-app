
import streamlit as st
import numpy as np


st.set_page_config(
    page_title="Have fun ",
    page_icon=":red_car:", 
)

import requests
import os
import time
import numpy as np
import matplotlib.pyplot as plt

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


def show_next(i, length):
  # Increments the counter to get next photo
  st.session_state[i] += 1
  if st.session_state[i] >= length:
    st.session_state[i] = 0
  

def show_last(i, length):
  # Decrements the counter to get next photo
  st.session_state[i] -= 1
  if st.session_state[i] < 0:
    st.session_state[i] = length-1


features = ["RAD", "RA", "RD", "AD", "spectrogram", "radarPC", "lidarPC", "image", "depth_image"]
show_status = [False, False, False, False, False, False, False, True, False]
# features = ["RAD", "image", "lidar"]
# show_status = [False, False, True]

expanders = [None]*len(features)
counters = [f"counter_{f}" for f in features]
placeholders = [[None]*6 for _ in range(len(features)) ]
load_actions = [None]*len(features)

# test expander with button 
for idx, (counter, feature) in enumerate(zip(counters, features)):
  expanders[idx] = st.expander(feature, expanded=True)
  with expanders[idx]:
    if counter not in st.session_state: 
      st.session_state[counter] = 0
    if feature not in st.session_state:
      st.session_state[feature] = show_status[idx]

    if st.session_state[feature]:

      fileset = [os.path.join(feature,f) for f in os.listdir(feature)]
      st.write(f"Total frames: {len(fileset)}")
      forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([counter,len(fileset)]), key=f"{feature}_forward_btn")
      backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([counter,len(fileset)]), key=f"{feature}_backward_btn")
      photo = fileset[st.session_state[counter]]
      st.image(photo,caption=photo)
      st.write(f"Index : {st.session_state[counter]}")
    else:

      for i in range(len(placeholders[idx])):
        placeholders[idx][i] = st.empty()

      load_actions[idx] = placeholders[idx][0].button("Get feature", key=f"{feature}_load_btn")
      if load_actions[idx]:
        placeholders[idx][0].empty()
        st.session_state[feature] = True
        with expanders[idx]:
          fileset = [os.path.join(feature,f) for f in os.listdir(feature)]
          #placeholder1.empty()
          placeholders[idx][1].write(f"Total frames: {len(fileset)}")
          forward_btn = placeholders[idx][2].button("Show next frame ⏭️",on_click=show_next,args=([counter,len(fileset)]), key=f"{feature}_forward_btn")
          backward_btn = placeholders[idx][3].button("Show last frame ⏪",on_click=show_last,args=([counter,len(fileset)]), key=f"{feature}_backward_btn")
          photo = fileset[st.session_state[counter]]
          placeholders[idx][4].image(photo,caption=photo)
          placeholders[idx][5].write(f"Index : {st.session_state[counter]}")
            

#st.info(f"load state: {st.session_state['load_state']}")

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
