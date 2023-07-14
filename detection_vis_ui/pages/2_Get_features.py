import streamlit as st
import requests
import os
import time
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Getfeature",
    page_icon=":red_car:", 
    layout="wide",
)

if 'datafile_chosen' not in st.session_state:
  st.info("Please choose a dataset first.")
  st.stop()


datafile_chosen = st.session_state.datafile_chosen
st.info(f"You have chosen {datafile_chosen['name']} data file.")
# st.json(datafile_chosen)

if 'datafile_parsed' not in st.session_state:
  st.session_state.datafile_parsed = False
  with st.spinner(text="Parsing data in progress..."):
    params = {
        "file_path": datafile_chosen["path"],
        "file_name": datafile_chosen["name"],
        "config": datafile_chosen["config"],
        "parser": datafile_chosen["parse"],
    }
    response = requests.get(f"http://detection_vis_backend:8001/parse", params=params)
  if response.status_code != 204:
    st.info("An error occurred in parsing data file.")
    st.stop()
  else:
    st.session_state.datafile_parsed = True


sources = ["RAD", "RA", "RD", "AD", "spectrogram", "radarPC", "lidarPC", "image", "depth_image"]
features = []
show_status = []
for idx, f in enumerate(sources):
  if datafile_chosen[f]:
    features.append(f)
    show_status.append(True)
  elif f in ("RAD", "RA", "RD", "AD", "spectrogram", "radarPC") and datafile_chosen["ADC"]:
    features.append(f)
    show_status.append(False)
  elif f in ("RA", "RD", "AD", "spectrogram", "radarPC") and "RAD" in features:
    features.append(f)
    show_status.append(False)

# st.info(features)
# st.info(status)


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


expanders = [None]*len(features)
counters = [f"counter_{f}" for f in features]
placeholders = [[None]*6 for _ in range(len(features)) ]
load_actions = [None]*len(features)
for idx, (counter, feature) in enumerate(zip(counters, features)):
  expanders[idx] = st.expander(feature, expanded=True)
  with expanders[idx]:
    if counter not in st.session_state: 
      st.session_state[counter] = 0
    if feature not in st.session_state:
      st.session_state[feature] = show_status[idx]
    
    params = {"parser": datafile_chosen["parse"]}
    
    if st.session_state[feature]:
      ###################### test code
      # fileset = [os.path.join("RAD",f) for f in os.listdir("RAD")]
      # st.write(f"Total frames: {len(fileset)}")
      # forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([counter,len(fileset)]), key=f"{feature}_forward_btn")
      # backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([counter,len(fileset)]), key=f"{feature}_backward_btn")
      # photo = fileset[st.session_state[counter]]
      # st.image(photo,caption=photo)

      response = requests.get(f"http://detection_vis_backend:8001/feature/{feature}/size", params=params)
      feature_size = response.json()
      st.write(f"Total frames: {feature_size}")
      forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([counter,feature_size]), key=f"{feature}_forward_btn")
      backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([counter,feature_size]), key=f"{feature}_backward_btn")
      params = {"parser": datafile_chosen["parse"]}
      response = requests.get(f"http://detection_vis_backend:8001/feature/{feature}/{st.session_state[counter]}", params=params)
      feature_data = response.json()
      
      if feature == "RAD":
        serialized_feature = feature_data["serialized_feature"]
        complex_feature = np.array([[[complex(real, imag) for real, imag in y] for y in z] for z in serialized_feature])
        feature_image = np.abs(complex_feature[:, 0, :])
        feature_image = feature_image - np.min(feature_image)
        feature_image = feature_image / np.max(feature_image)
        st.image(feature_image, caption=f"{feature_image.shape}")
      else:
        serialized_feature = feature_data["serialized_feature"]
        feature_image = np.array(serialized_feature)
        feature_image = feature_image - np.min(feature_image)
        feature_image = feature_image / np.max(feature_image)
        st.image(feature_image, caption=f"{feature_image.shape}")

      st.write(f"Index : {st.session_state[counter]}")
    else:
      for i in range(len(placeholders[idx])):
        placeholders[idx][i] = st.empty()

      load_actions[idx] = placeholders[idx][0].button("Get feature", key=f"{feature}_load_btn")
     
      if load_actions[idx]:
        
        placeholders[idx][0].empty()
        st.session_state[feature] = True
        with expanders[idx]:
          #######################test code
          # fileset = [os.path.join("RAD",f) for f in os.listdir("RAD")]
          # placeholders[idx][1].write(f"Total frames: {len(fileset)}")
          # forward_btn = placeholders[idx][2].button("Show next frame ⏭️",on_click=show_next,args=([counter,len(fileset)]), key=f"{feature}_forward_btn")
          # backward_btn = placeholders[idx][3].button("Show last frame ⏪",on_click=show_last,args=([counter,len(fileset)]), key=f"{feature}_backward_btn")
          # photo = fileset[st.session_state[counter]]
          # placeholders[idx][4].image(photo,caption=photo) 

          with st.spinner(text="Getting the feature in progress..."):
            response = requests.get(f"http://detection_vis_backend:8001/feature/{feature}/size", params=params)
            feature_size = response.json()
          
          placeholders[idx][1].write(f"Total frames: {feature_size}")
          forward_btn = placeholders[idx][2].button("Show next frame ⏭️",on_click=show_next,args=([counter,feature_size]), key=f"{feature}_forward_btn")
          backward_btn = placeholders[idx][3].button("Show last frame ⏪",on_click=show_last,args=([counter,feature_size]), key=f"{feature}_backward_btn")
          response = requests.get(f"http://detection_vis_backend:8001/feature/{feature}/{st.session_state[counter]}", params=params)
          feature_data = response.json()
          if feature == "RAD":
            serialized_feature = feature_data["serialized_feature"]
            complex_feature = np.array([[[complex(real, imag) for real, imag in y] for y in z] for z in serialized_feature])
            feature_image = np.abs(complex_feature[:, 0, :])
            feature_image = feature_image - np.min(feature_image)
            feature_image = feature_image / np.max(feature_image)
            placeholders[idx][4].image(feature_image, caption=f"{feature_image.shape}")
          else:
            serialized_feature = feature_data["serialized_feature"]
            feature_image = np.array(serialized_feature)
            feature_image = feature_image - np.min(feature_image)
            feature_image = feature_image / np.max(feature_image)
            placeholders[idx][4].image(feature_image, caption=f"{feature_image.shape}")
          placeholders[idx][5].write(f"Index : {st.session_state[counter]}")







# def show_next_spectrogram(length):
#   # Increments the counter to get next photo
#   st.session_state.counter_spectrogram += 1
#   if st.session_state.counter_spectrogram >= length:
#     st.session_state.counter_spectrogram = 0
  

# def show_last_spectrogram(length):
#   # Decrements the counter to get next photo
#   st.session_state.counter_spectrogram -= 1
#   if st.session_state.counter_spectrogram < 0:
#     st.session_state.counter_spectrogram = length-1


# def show_next_image(length):
#   # Increments the counter to get next photo
#   st.session_state.counter_image += 1
#   if st.session_state.counter_image >= length:
#     st.session_state.counter_image = 0
  

# def show_last_image(length):
#   # Decrements the counter to get next photo
#   st.session_state.counter_image -= 1
#   if st.session_state.counter_image < 0:
#     st.session_state.counter_image = length-1


# def make_grid(cols,rows):
#   grid = [0]*cols
#   for i in range(cols):
#     with st.container():
#       grid[i] = st.columns(rows)
#   return grid


# # layout of the page
# grid = make_grid(2,1)

# # datafile_chosen = {}
# # datafile_chosen["spectrogram"] = True
# # datafile_chosen["image"] = True

# if datafile_chosen["spectrogram"]:
#   grid[0][0].header(f"Spectrogram feature")

#   col1_spectrogram,col2_spectrogram = grid[0][0].columns(2)

#   if 'counter_spectrogram' not in st.session_state: 
#     st.session_state.counter_spectrogram = 0

#   # Get list of images in folder
#   feature_spectrogram_subpath = r"spectrograms"
#   feature_spectrogram_set = [os.path.join(feature_spectrogram_subpath,f) for f in os.listdir(feature_spectrogram_subpath)]
#   col1_spectrogram.write(f"Total frames: {len(feature_spectrogram_set)}")
#   col1_spectrogram.write(feature_spectrogram_set)

#   forward_spec_btn = col1_spectrogram.button("Show next frame ⏭️",on_click=show_next_spectrogram,args=([len(feature_spectrogram_set)]), key="spectrogram_forward_btn")
#   backward_spec_btn = col1_spectrogram.button("Show last frame ⏪",on_click=show_last_spectrogram,args=([len(feature_spectrogram_set)]), key="spectrogram_backward_btn")
#   photo = feature_spectrogram_set[st.session_state.counter_spectrogram]
#   col2_spectrogram.image(photo,caption=photo)
#   col1_spectrogram.write(f"Index : {st.session_state.counter_spectrogram}")


# # show camera images
# if datafile_chosen["image"]:
#   grid[1][0].header(f"Camera images")

#   col1_image,col2_image = grid[1][0].columns(2)

#   if 'counter_image' not in st.session_state: 
#     st.session_state.counter_image = 0

#   # Get list of images in folder
#   params = {
#     "file_path": datafile_chosen["parse"],
#   }
#   response = requests.get(f"http://detection_vis_backend:8001/feature/image/0", params=params)
#   feature = response.json()
#   photo = np.array(feature)
#   # photo = photo - np.min(photo)
#   # photo = photo / np.max(photo)

#   fig, ax = plt.subplots(figsize=(5, 5))
#   ax.imshow(photo, aspect='auto')
#   col2_image.pyplot(fig, use_container_width=False)

#   # feature_subpath = r"images"
#   # feature_set = [os.path.join(feature_subpath,f) for f in os.listdir(feature_subpath)]
#   # col1_image.write(f"Total frames: {len(feature_set)}")
#   # col1_image.write(feature_set)
#   # forward_imag_btn = col1_image.button("Show next frame ⏭️",on_click=show_next_image,args=([len(feature_set)]), key="image_forward_btn")
#   # backward_imag_btn = col1_image.button("Show last frame ⏪",on_click=show_last_image,args=([len(feature_set)]), key="image_backward_btn")
#   # photo = feature_set[st.session_state.counter_image]
#   #col2_image.image(photo,caption=photo)
#   # col1_image.write(f"Index : {st.session_state.counter_image}")
#   col1_image.write(f"Index : ")

