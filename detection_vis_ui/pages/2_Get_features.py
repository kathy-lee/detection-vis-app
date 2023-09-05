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


featureset = ["ADC", "RAD", "RA", "RD", "spectrogram", "radarPC", "lidarPC", "image", "depth_image"]

# frame sync mode
if 'frame_sync' not in st.session_state:
  st.session_state.frame_sync = False

if 'frame_id' not in st.session_state:
  st.session_state.frame_id = 0

def reset_file():
  st.session_state.frame_id = 0
  st.session_state.frame_sync = False
  if 'sync_checkbox' in st.session_state:
    st.session_state.sync_checkbox = False
  if 'frame_slider' in st.session_state:
    st.session_state.frame_slider = 0
  for feature in featureset:
    counter = f"counter_{feature}"
    if counter in st.session_state:
      st.session_state[counter] = 0

# old
# datafile_name = st.selectbox("Which data file would you like to check the features?", [f["name"] for f in datafiles_chosen], 
#                              index=st.session_state.datafile_chosen)
# st.session_state.datafile_chosen = next((index for (index, d) in enumerate(datafiles_chosen) if d["name"] == datafile_name), None)

##### new
default_index = 0 if st.session_state['datafile_chosen'] > len(st.session_state['datafiles_chosen'])-1 else st.session_state['datafile_chosen']
st.selectbox("Which data file would you like to check the features?", [f["name"] for f in datafiles_chosen], 
                             index=default_index, key='datafile_selectbox', on_change=reset_file)
st.session_state.datafile_chosen = next((index for (index, d) in enumerate(datafiles_chosen) 
                                         if d["name"] == st.session_state['datafile_selectbox']), None)
#### 

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


# # frame sync mode
# if 'frame_sync' not in st.session_state:
#   st.session_state.frame_sync = False

@st.cache_data
def check_sync(file_id):
  response = requests.get(f"http://{backend_service}:8001/sync_check/{file_id}")
  sync = response.json()
  return sync

# @st.cache_data
def set_sync(file_id):
  response = requests.get(f"http://{backend_service}:8001/sync_set/{file_id}")
  frames = response.json()
  return frames

# @st.cache_data
def unset_sync(file_id):
  response = requests.get(f"http://{backend_service}:8001/sync_unset/{file_id}")
  return True if response.status_code == 204 else False

sync_frames = check_sync(datafile_chosen["id"])
st.write(f"sync_frames: {sync_frames}")
st.session_state.frame_id = 0

def set_sync_mode():
  st.session_state.frame_sync = st.session_state['sync_checkbox']
  return 

st.checkbox("frame sync mode", value=st.session_state.frame_sync, key='sync_checkbox', on_change=set_sync_mode)
if st.session_state['sync_checkbox']:
  # st.session_state.frame_sync = True
  if sync_frames == 0:
    sync_frames = set_sync(datafile_chosen['id'])
  
  st.slider('Choose a frame', 0, sync_frames, 0, key='frame_slider')
else:
  # st.session_state.frame_sync = False
  if sync_frames == 0:
    unset_sync(datafile_chosen['id'])
  

# infer the available features
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
    features_show.append(False)
    # if f != "RAD":
    #   features_show.append(False)
    # else:
    #   features_show.append(True) # Do not support 3d array display
  elif f in ("RA", "RD", "spectrogram", "radarPC") and "RAD" in features:
    features.append(f)
    features_show.append(False)

# for debug
st.info(features)
st.info(features_show)

@st.cache_data
def get_feature(file_id, feature, idx):
  response = requests.get(f"http://{backend_service}:8001/feature/{file_id}/{feature}/{idx}")
  feature_data = response.json()
  serialized_feature = feature_data["serialized_feature"]
  return serialized_feature

@st.cache_data
def get_feature_size(file_id, feature):
  response = requests.get(f"http://{backend_service}:8001/feature/{file_id}/{feature}/size")
  feature_size = response.json()
  return feature_size

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


def show_feature(file_id, feature, counter, frame_id, config=None):
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

  feature_size = get_feature_size(file_id, feature)
  st.write(f"Total frames: {feature_size}")
  forward_btn = st.button("Show next frame ⏭️",on_click=show_next,args=([counter,feature_size]), 
                          key=f"{feature}_forward_btn", disabled=st.session_state.frame_sync)
  backward_btn = st.button("Show last frame ⏪",on_click=show_last,args=([counter,feature_size]), 
                           key=f"{feature}_backward_btn", disabled=st.session_state.frame_sync)
  if st.session_state.frame_sync:
    st.session_state[counter] = frame_id

  serialized_feature = get_feature(file_id, feature, st.session_state[counter])
  feature_image = np.array(serialized_feature)
  if feature == "lidarPC" or feature == "radarPC":
    plt.figure(figsize=(8, 6))
    plt.plot(feature_image[:,1], feature_image[:,0], '.')
    plt.xlim(-max(abs(feature_image[:,1])), max(abs(feature_image[:,1]))) # plt.xlim(-20,20)
    plt.ylim(0, max(feature_image[:,0])) # plt.ylim(0,100)
    plt.grid()
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"index: {st.session_state[counter]}, shape: {feature_image.shape}", y=-0.1)
    st.pyplot(plt)
  elif feature == "RD":
    #rangedoppler = feature_image[...,::2] + 1j * feature_image[...,1::2]
    #power_spectrum = np.sum(np.abs(rangedoppler),axis=2)
    #power_spectrum = np.abs(rangedoppler)
    #plt.figure(figsize=(8,6))
    #plt.imshow(np.log10(power_spectrum), aspect='auto')
    plt.figure(figsize=(8,6))
    plt.imshow(feature_image) #, aspect='auto'
    plt.title(f"index: {st.session_state[counter]}, shape: {feature_image.shape}", y=-0.1)
    st.pyplot(plt)
  elif feature == "RA":
    if feature_image.ndim > 2:
      feature_image = np.sqrt(feature_image[:, :, 0] ** 2 + feature_image[:, :, 1] ** 2)
    plt.figure(figsize=(8,6))
    plt.imshow(feature_image, aspect='auto')
    plt.title(f"index: {st.session_state[counter]}, shape: {feature_image.shape}", y=-0.1)
    st.pyplot(plt)
  else:
    feature_image = feature_image - np.min(feature_image)
    feature_image = feature_image / np.max(feature_image)
    st.image(feature_image, caption=f"index: {st.session_state[counter]}, shape: {feature_image.shape}")
  return


if 'frame_slider' in st.session_state:
  st.session_state.frame_id = st.session_state['frame_slider']


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


feature = "RA"
counter = f"counter_{feature}" 
if feature in features:
  if not features_show[features.index(feature)]:
    if "aoa_cfg" not in st.session_state:
      st.session_state.aoa_cfg = 0
    aoa_config_list = ["Not interested", "Barlett", "Capon"]
    aoa_config = st.sidebar.radio("How would you like to do AoA estimation?", aoa_config_list, index=st.session_state.aoa_cfg)
    st.session_state.aoa_cfg = aoa_config_list.index(aoa_config)
    #
    if aoa_config in ("Barlett", "Capon"):
      expander_RA = st.expander("Range-Azimuth(RA) feature", expanded=True)
      if counter not in st.session_state:
        st.session_state[counter] = st.session_state.frame_id
      with expander_RA:
        show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id, config=aoa_config)
  else:
    expander_RA = st.expander("Range-Azimuth(RA) feature", expanded=True)
    if counter not in st.session_state:
      st.session_state[counter] = st.session_state.frame_id
    with expander_RA:
      show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id, config=None)


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







feature = "image"
counter = f"counter_{feature}" 
if feature in features:
  if counter not in st.session_state:
    st.session_state[counter] = st.session_state.frame_id
  expander_image = st.expander("RGB images", expanded=True)
  with expander_image:
    show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id)
    

feature = "depth_image"
counter = f"counter_{feature}" 
if feature in features:
  if counter not in st.session_state:
    st.session_state[counter] = st.session_state.frame_id
  expander_depthimage = st.expander("Depth images", expanded=True)
  with expander_depthimage:
    show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id)


feature = "lidarPC"
counter = f"counter_{feature}" 
if feature in features:
  if counter not in st.session_state:
    st.session_state[counter] = st.session_state.frame_id
  expander_lidarpc = st.expander("Lidar Point Cloud", expanded=True)
  with expander_lidarpc:
    show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id)


feature = "RD"
counter = f"counter_{feature}" 
if feature in features and (features_show[features.index("RD")] or fft_config in ("No windowing", "Hamming windowing", "Hanning windowing")):
  expander_RD = st.expander("Range-Doppler(RD) feature", expanded=True)
  if counter not in st.session_state: 
    st.session_state[counter] = st.session_state.frame_id
  with expander_RD:
    cfg = None if features_show[features.index("RD")] else fft_config
    show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id, config=cfg)


# feature = "RA" 
# counter = f"counter_{feature}"  
# if aoa_config in ("Barlett", "Capon"):
#   expander_RA = st.expander("Range-Azimuth(RA) feature", expanded=True)
#   if counter not in st.session_state:
#     st.session_state[counter] = st.session_state.frame_id
#   with expander_RA:
#     show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id, config=aoa_config)
# elif feature in features:
#   expander_RA = st.expander("Range-Azimuth(RA) feature", expanded=True)
#   if counter not in st.session_state:
#     st.session_state[counter] = st.session_state.frame_id
#   with expander_RA:
#     show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id, config=None)


feature = "spectrogram" 
counter = f"counter_{feature}"  
if feature in features and tfa_config in ("STFT", "WV"):
  expander_tfa = st.expander("Spectrogram feature", expanded=True)
  if counter not in st.session_state:
    st.session_state[counter] = st.session_state.frame_id
  with expander_tfa:
    show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id, config=tfa_config)


feature = "radarPC" 
counter = f"counter_{feature}"  
if feature in features and (features_show[features.index("radarPC")] or cfar_config in ("CA-CFAR", "CASO-CFAR", "CAGO-CFAR", "OS-CFAR")):
  expander_radarpc = st.expander("Radar Point Cloud", expanded=True)
  if counter not in st.session_state:
    st.session_state[counter] = st.session_state.frame_id
  with expander_radarpc:
    cfg = None if features_show[features.index("radarPC")] else cfar_config
    show_feature(datafile_chosen['id'], feature, counter, st.session_state.frame_id, config=cfg)


@st.cache_data
def get_feature_video(file_id, features):
  param = {"feature_list": features}
  response = requests.get(f"http://{backend_service}:8001/video/{file_id}", params=param)
  video_path = response.json()
  return video_path

# get current shown features' names
feature_list = [f for f, b in zip(features, features_show) if b]
if "cfar_cfg" in st.session_state:
  if st.session_state.cfar_cfg > 0 and 'radarPC' not in feature_list:
    feature_list.append('radarPC')
if "aoa_cfg" in st.session_state:
  if st.session_state.aoa_cfg > 0 and 'RA' not in feature_list:
    feature_list.append('RA')
if "tfa_cfg" in st.session_state:
  if st.session_state.tfa_cfg > 0 and 'spectrogram' not in feature_list:
    feature_list.append('spectrogram')
if "fft_cfg" in st.session_state:
  if st.session_state.fft_cfg > 0 and 'RD' not in feature_list:
    feature_list.append('RD')

# ###### old video widget
# if st.session_state.frame_sync:
#   if 'video_features' not in st.session_state:
#     video_feature_list_val = st.multiselect("Ajust the features you would like to autodisplay:", feature_list, feature_list, key="video_features")
#   else:
#     video_feature_list_val = st.multiselect("Ajust the features you would like to autodisplay:", feature_list, st.session_state['video_features'], key="video_features")
#   auto_display = st.checkbox("Auto display")
#   if auto_display:
#     video_path = get_feature_video(datafile_chosen['id'], st.session_state['video_features'])
#     # param = {"feature_list": feature_list}
#     # response = requests.get(f"http://{backend_service}:8001/video/{datafile_chosen['id']}", params=param)
#     # video_path = response.json()
#     st.video(video_path)
# ######

##### new video widget
if 'video_features' not in st.session_state or set(st.session_state.video_features) - set(feature_list):
  st.session_state.video_features = feature_list

def set_video_features():
  st.session_state['video_features'] = st.session_state['video_features_selectbox']

if st.session_state.frame_sync:
  st.multiselect("Ajust the features you would like to autodisplay:", feature_list, st.session_state['video_features'], 
                 key="video_features_selectbox", on_change=set_video_features)
  auto_display = st.checkbox("Auto display")
  if auto_display:
    video_path = get_feature_video(datafile_chosen['id'], st.session_state['video_features'])
    # param = {"feature_list": feature_list}
    # response = requests.get(f"http://{backend_service}:8001/video/{datafile_chosen['id']}", params=param)
    # video_path = response.json()
    st.video(video_path)
#####


##### old train feature wideget
# if 'features_chosen' not in st.session_state:
#   st.write("'features_chosen' not in st.session_state")
#   features_chosen_val = st.multiselect("Which features would you like to select as train input?", features, key="features_chosen")
# else:
#   features_chosen_val = st.multiselect("Which features would you like to select as train input?", features, st.session_state['features_chosen'], key="features_chosen")

# button_click = st.button("Go to train")
# if button_click:
#   if features_chosen_val:
#     #check_datafiles(st.session_state.datafiles_chosen)
#     switch_page("train model")
#   else:
#     st.info("Please choose features first")
######


##### new train feature wideget
if 'features_chosen' not in st.session_state:
  st.session_state['features_chosen'] = None

def set_train_features():
  st.session_state['features_chosen'] = st.session_state['train_features_selectbox']

st.multiselect("Which features would you like to select as train input?", features, st.session_state['features_chosen'], 
               key="train_features_selectbox", on_change=set_train_features)

button_click = st.button("Go to train", key='switch_train_page')
if button_click:
  if st.session_state.features_chosen:
    #check_datafiles(st.session_state.datafiles_chosen)
    switch_page("train model")
  else:
    st.info("Please choose features first")
######


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

