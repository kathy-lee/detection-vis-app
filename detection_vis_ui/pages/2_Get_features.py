import streamlit as st
import requests
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from streamlit_extras.switch_page_button import switch_page



backend_service = os.getenv('BACKEND_SERVICE', 'localhost')

st.set_page_config(
    page_title="Getfeature",
    page_icon=":red_car:", 
)


if 'datafiles_chosen' not in st.session_state or not st.session_state.datafiles_chosen:
  st.info("Please choose the data first.")
  st.stop()

datafiles_chosen = st.session_state.datafiles_chosen

if 'datafile_chosen' not in st.session_state:
  st.session_state.datafile_chosen = 0

datafile_name = st.selectbox("Which data file would you like to check the features?", [f["name"] for f in datafiles_chosen], 
                             index=st.session_state.datafile_chosen)
st.session_state.datafile_chosen = next((index for (index, d) in enumerate(datafiles_chosen) if d["name"] == datafile_name), None)

# get the chosen data file
datafile_chosen = {}
#datafile_chosen = next(f for f in datafiles_chosen if f["name"] == datafile_name)
datafile_chosen = datafiles_chosen[st.session_state.datafile_chosen]
#st.write(datafile_chosen)


# parse the chosen data file
datafile_parsed = f"parse_{datafile_chosen['id']}"
if datafile_parsed not in st.session_state:
  st.session_state[datafile_parsed] = False
  with st.spinner(text="Parsing data file in progress..."):
    response = requests.get(f"http://{backend_service}:8001/parse/{datafile_chosen['id']}")
  if response.status_code != 204:
    st.info("An error occurred in parsing data file.")
    st.stop()
  else:
    st.session_state[datafile_parsed] = True


# frame sync mode
if 'frame_sync' not in st.session_state:
  st.session_state.frame_sync = False

response = requests.get(f"http://{backend_service}:8001/sync/{datafile_chosen['id']}")
sync = response.json()
frame_sync = st.checkbox("frame sync mode", value=st.session_state.frame_sync, disabled=not sync)
frame_begin = 0
frame_end = sync
frame_id = 0
if frame_sync:
  st.session_state.frame_sync = True
  frame_id = st.slider('Choose a frame', frame_begin, frame_end, frame_begin)
else:
  #st.slider('Choose a frame', frame_begin, frame_end, frame_begin, disabled=True)
  st.session_state.frame_sync = False
  

# infer the available features
featureset = ["ADC", "RAD", "RA", "RD", "spectrogram", "radarPC", "lidarPC", "image", "depth_image"]
features = []
features_show = []
for idx, f in enumerate(featureset):
  if datafile_chosen[f]:
    features.append(f)
    if f in ("ADC", "RAD"):
      features_show.append(False)
    else:
      features_show.append(True)
  elif f in ("RAD", "RA", "RD", "spectrogram", "radarPC") and datafile_chosen["ADC"]:
    features.append(f)
    if f != "RAD":
      features_show.append(False)
    else:
      features_show.append(True)
  elif f in ("RA", "RD", "spectrogram", "radarPC") and "RAD" in features:
    features.append(f)
    features_show.append(False)

st.info(features)
st.info(features_show)


if "RD" in features and not features_show[features.index("RD")]:
  if "fft_cfg" not in st.session_state:
    st.session_state.fft_cfg = 0
  fft_config_list = ["Not interested", "No windowing", "Hamming windowing", "Hanning windowing"]
  fft_config = st.sidebar.radio("How would you like to do Range&Doppler-FFT?", fft_config_list, index = st.session_state.fft_cfg)
  st.session_state.fft_cfg = fft_config_list.index(fft_config)


if "spectrogram" in features and not features_show[features.index("spectrogram")]:
  if "tfa_cfg" not in st.session_state:
    st.session_state.tfa_cfg = 0
  tfa_config_list = ["Not interested", "STFT", "WV"]
  tfa_config = st.sidebar.radio("How would you like to do time frequency analysis?", tfa_config_list, index=st.session_state.tfa_cfg)
  st.session_state.tfa_cfg = tfa_config_list.index(tfa_config)


if "RA" in features and not features_show[features.index("RA")]:
  if "aoa_cfg" not in st.session_state:
    st.session_state.aoa_cfg = 0
  aoa_config_list = ["Not interested", "Barlett", "Capon"]
  aoa_config = st.sidebar.radio("How would you like to do AoA estimation?", aoa_config_list, index=st.session_state.aoa_cfg)
  st.session_state.aoa_cfg = aoa_config_list.index(aoa_config)


if "noise_cfg" not in st.session_state:
  st.session_state.noise_cfg = 0
noise_config_list = ["Not interested", "Static noise removal"]
noise_config = st.sidebar.radio("How would you like to remove nosie?", noise_config_list, index=st.session_state.noise_cfg)
st.session_state.noise_cfg = noise_config_list.index(noise_config)


if "radarPC" in features and not features_show[features.index("radarPC")]:
  if "cfar_cfg" not in st.session_state:
    st.session_state.cfar_cfg = 0
  cfar_config_list = ["Not interested", "CA-CFAR", "CASO-CFAR", "CAGO-CFAR", "OS-CFAR"]
  cfar_config = st.sidebar.radio("How would you like to do CFAR detection?", cfar_config_list, index=st.session_state.cfar_cfg)
  st.session_state.cfar_cfg = cfar_config_list.index(cfar_config)


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

@st.cache_data(experimental_allow_widgets=True)
def show_feature(feature, counter, frame_id, config=None):
  ###################### test code
  # fileset = [os.path.join("detection_vis_ui/image",f) for f in os.listdir("detection_vis_ui/image")]
  # st.write(f"Total frames: {len(fileset)}")
  # forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([counter,len(fileset)]), 
  #                         key=f"{feature}_forward_btn", disabled=st.session_state.frame_sync)
  # backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([counter,len(fileset)]), 
  #                         key=f"{feature}_backward_btn", disabled=st.session_state.frame_sync)
  # if st.session_state.frame_sync:
  #   st.session_state[counter] = frame_id
  # photo = fileset[st.session_state[counter]]
  # st.image(photo,caption=st.session_state[counter])


  response = requests.get(f"http://{backend_service}:8001/feature/{datafile_chosen['id']}/{feature}/size")
  feature_size = response.json()
  st.write(f"Total frames: {feature_size}")
  forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([counter,feature_size]), 
                          key=f"{feature}_forward_btn", disabled=st.session_state.frame_sync)
  backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([counter,feature_size]), 
                           key=f"{feature}_backward_btn", disabled=st.session_state.frame_sync)
  if st.session_state.frame_sync:
    st.session_state[counter] = frame_id

  response = requests.get(f"http://{backend_service}:8001/feature/{datafile_chosen['id']}/{feature}/{st.session_state[counter]}")
  feature_data = response.json()
  serialized_feature = feature_data["serialized_feature"]
  feature_image = np.array(serialized_feature)
  if feature == "lidarPC" or feature == "radarPC":
    plt.figure(figsize=(8, 6))
    plt.plot(feature_image[:,1], feature_image[:,0], '.')
    plt.xlim(-20,20)
    plt.ylim(0,100)
    plt.grid()
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"index: {st.session_state[counter]}", y=-0.1)
    st.pyplot(plt)
  elif feature == "RD":
    rangedoppler = feature_image[...,::2] + 1j * feature_image[...,1::2]
    power_spectrum = np.sum(np.abs(rangedoppler),axis=2)
    plt.figure(figsize=(8,6))
    plt.imshow(np.log10(power_spectrum), aspect='auto')
    plt.title(f"index: {st.session_state[counter]}", y=-0.1)
    st.pyplot(plt)
  else:
    feature_image = feature_image - np.min(feature_image)
    feature_image = feature_image / np.max(feature_image)
    st.image(feature_image, caption=f"index: {st.session_state[counter]}")
  
  # if feature == "RAD":
  #   serialized_feature = feature_data["serialized_feature"]
  #   complex_feature = np.array([[[complex(real, imag) for real, imag in y] for y in z] for z in serialized_feature])
  #   feature_image = np.abs(complex_feature[:, 0, :])
  #   feature_image = feature_image - np.min(feature_image)
  #   feature_image = feature_image / np.max(feature_image)
  #   st.image(feature_image, caption=f"{feature_image.shape}")
  # else:
  #   serialized_feature = feature_data["serialized_feature"]
  #   feature_image = np.array(serialized_feature)
  #   feature_image = feature_image - np.min(feature_image)
  #   feature_image = feature_image / np.max(feature_image)
  #   st.image(feature_image, caption=f"{feature_image.shape}")

  # st.write(f"Index : {st.session_state[counter]}")


feature = "image"
counter = f"counter_{feature}" 
if feature in features:
  if counter not in st.session_state:
    st.session_state[counter] = frame_id
  expander_image = st.expander("RGB images", expanded=True)
  with expander_image:
    show_feature(feature, counter, frame_id)
    

feature = "depth_image"
counter = f"counter_{feature}" 
if feature in features:
  if counter not in st.session_state:
    st.session_state[counter] = frame_id
  expander_depthimage = st.expander("Depth images", expanded=True)
  with expander_depthimage:
    show_feature(feature, counter, frame_id)


feature = "lidarPC"
counter = f"counter_{feature}" 
if feature in features:
  if counter not in st.session_state:
    st.session_state[counter] = frame_id
  expander_lidarpc = st.expander("Lidar Point Cloud", expanded=True)
  with expander_lidarpc:
    show_feature(feature, counter, frame_id)


feature = "RD"
counter = f"counter_{feature}" 
if feature in features and (features_show[features.index("RD")] or fft_config in ("No windowing", "Hamming windowing", "Hanning windowing")):
  expander_RD = st.expander("Range-Doppler(RD) feature", expanded=True)
  if counter not in st.session_state: 
    st.session_state[counter] = frame_id
  with expander_RD:
    cfg = None if features_show[features.index("RD")] else fft_config
    show_feature(feature, counter, frame_id, config=cfg)


feature = "RA" 
counter = f"counter_{feature}"  
if feature in features and aoa_config in ("Barlett", "Capon"):
  expander_RA = st.expander("Range-Azimuth(RA) feature", expanded=True)
  if counter not in st.session_state:
    st.session_state[counter] = frame_id
  with expander_RA:
    show_feature(feature, counter, frame_id, config=aoa_config)


feature = "spectrogram" 
counter = f"counter_{feature}"  
if feature in features and tfa_config in ("STFT", "WV"):
  expander_tfa = st.expander("Spectrogram feature", expanded=True)
  if counter not in st.session_state:
    st.session_state[counter] = frame_id
  with expander_tfa:
    show_feature(feature, counter, frame_id, config=tfa_config)


feature = "radarPC" 
counter = f"counter_{feature}"  
if feature in features and (features_show[features.index("radarPC")] or cfar_config in ("CA-CFAR", "CASO-CFAR", "CAGO-CFAR", "OS-CFAR")):
  expander_radarpc = st.expander("Radar Point Cloud", expanded=True)
  if counter not in st.session_state:
    st.session_state[counter] = frame_id
  with expander_radarpc:
    cfg = None if features_show[features.index("radarPC")] else cfar_config
    show_feature(feature, counter, frame_id, config=cfg)


if 'features_chosen' not in st.session_state:
  st.session_state.features_chosen = []

features_chosen = st.multiselect("Which features would you like to select as input?", features, st.session_state.features_chosen)
st.session_state.features_chosen = features_chosen
st.write(features_chosen)

button_click = st.button("Go to train")
if button_click:
  if features_chosen:
    #check_datafiles(st.session_state.datafiles_chosen)
    switch_page("train model")
  else:
    st.info("Please choose features first")

# #############################################
# expanders = [None]*len(features)
# counters = [f"counter_{f}" for f in features]
# placeholders = [[None]*6 for _ in range(len(features)) ]
# load_actions = [None]*len(features)
# for idx, (counter, feature) in enumerate(zip(counters, features)):
#   expanders[idx] = st.expander(feature, expanded=True)
#   with expanders[idx]:
#     if counter not in st.session_state: 
#       st.session_state[counter] = 0
#     if feature not in st.session_state:
#       st.session_state[feature] = show_status[idx]
    
#     params = {"parser": datafile_chosen["parse"]}
    
#     if st.session_state[feature]:
#       ###################### test code
#       # fileset = [os.path.join("RAD",f) for f in os.listdir("RAD")]
#       # st.write(f"Total frames: {len(fileset)}")
#       # forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([counter,len(fileset)]), key=f"{feature}_forward_btn")
#       # backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([counter,len(fileset)]), key=f"{feature}_backward_btn")
#       # photo = fileset[st.session_state[counter]]
#       # st.image(photo,caption=photo)

#       response = requests.get(f"http://{backend_service}:8001/feature/{feature}/size", params=params)
#       feature_size = response.json()
#       st.write(f"Total frames: {feature_size}")
#       forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([counter,feature_size]), key=f"{feature}_forward_btn")
#       backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([counter,feature_size]), key=f"{feature}_backward_btn")
#       params = {"parser": datafile_chosen["parse"]}
#       response = requests.get(f"http://{backend_service}:8001/feature/{feature}/{st.session_state[counter]}", params=params)
#       feature_data = response.json()
      
#       if feature == "RAD":
#         serialized_feature = feature_data["serialized_feature"]
#         complex_feature = np.array([[[complex(real, imag) for real, imag in y] for y in z] for z in serialized_feature])
#         feature_image = np.abs(complex_feature[:, 0, :])
#         feature_image = feature_image - np.min(feature_image)
#         feature_image = feature_image / np.max(feature_image)
#         st.image(feature_image, caption=f"{feature_image.shape}")
#       else:
#         serialized_feature = feature_data["serialized_feature"]
#         feature_image = np.array(serialized_feature)
#         feature_image = feature_image - np.min(feature_image)
#         feature_image = feature_image / np.max(feature_image)
#         st.image(feature_image, caption=f"{feature_image.shape}")

#       st.write(f"Index : {st.session_state[counter]}")
#     else:
#       for i in range(len(placeholders[idx])):
#         placeholders[idx][i] = st.empty()

#       load_actions[idx] = placeholders[idx][0].button("Get feature", key=f"{feature}_load_btn")
     
#       if load_actions[idx]:
        
#         placeholders[idx][0].empty()
#         st.session_state[feature] = True
#         with expanders[idx]:
#           #######################test code
#           # fileset = [os.path.join("RAD",f) for f in os.listdir("RAD")]
#           # placeholders[idx][1].write(f"Total frames: {len(fileset)}")
#           # forward_btn = placeholders[idx][2].button("Show next frame ⏭️",on_click=show_next,args=([counter,len(fileset)]), key=f"{feature}_forward_btn")
#           # backward_btn = placeholders[idx][3].button("Show last frame ⏪",on_click=show_last,args=([counter,len(fileset)]), key=f"{feature}_backward_btn")
#           # photo = fileset[st.session_state[counter]]
#           # placeholders[idx][4].image(photo,caption=photo) 

#           with st.spinner(text="Getting the feature in progress..."):
#             response = requests.get(f"http://{backend_service}:8001/feature/{feature}/size", params=params)
#             feature_size = response.json()
          
#           placeholders[idx][1].write(f"Total frames: {feature_size}")
#           forward_btn = placeholders[idx][2].button("Show next frame ⏭️",on_click=show_next,args=([counter,feature_size]), key=f"{feature}_forward_btn")
#           backward_btn = placeholders[idx][3].button("Show last frame ⏪",on_click=show_last,args=([counter,feature_size]), key=f"{feature}_backward_btn")
#           response = requests.get(f"http://{backend_service}:8001/feature/{feature}/{st.session_state[counter]}", params=params)
#           feature_data = response.json()
#           if feature == "RAD":
#             serialized_feature = feature_data["serialized_feature"]
#             complex_feature = np.array([[[complex(real, imag) for real, imag in y] for y in z] for z in serialized_feature])
#             feature_image = np.abs(complex_feature[:, 0, :])
#             feature_image = feature_image - np.min(feature_image)
#             feature_image = feature_image / np.max(feature_image)
#             placeholders[idx][4].image(feature_image, caption=f"{feature_image.shape}")
#           else:
#             serialized_feature = feature_data["serialized_feature"]
#             feature_image = np.array(serialized_feature)
#             feature_image = feature_image - np.min(feature_image)
#             feature_image = feature_image / np.max(feature_image)
#             placeholders[idx][4].image(feature_image, caption=f"{feature_image.shape}")
#           placeholders[idx][5].write(f"Index : {st.session_state[counter]}")







################################### Grid layout

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

