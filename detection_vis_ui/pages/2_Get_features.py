import streamlit as st
import requests
import os
import time

st.set_page_config(
    page_title="Getfeature",
    page_icon=":red_car:", 
    layout="wide",
)


datafile_chosen = st.session_state.datafile_chosen
st.info(f"You have chosen {datafile_chosen['name']} data file.")
# st.json(datafile_chosen)

def show_next_spectrogram(length):
  # Increments the counter to get next photo
  st.session_state.counter_spectrogram += 1
  if st.session_state.counter_spectrogram >= length:
    st.session_state.counter_spectrogram = 0
  

def show_last_spectrogram(length):
  # Decrements the counter to get next photo
  st.session_state.counter_spectrogram -= 1
  if st.session_state.counter_spectrogram < 0:
    st.session_state.counter_spectrogram = length-1


def show_next_image(length):
  # Increments the counter to get next photo
  st.session_state.counter_image += 1
  if st.session_state.counter_image >= length:
    st.session_state.counter_image = 0
  

def show_last_image(length):
  # Decrements the counter to get next photo
  st.session_state.counter_image -= 1
  if st.session_state.counter_image < 0:
    st.session_state.counter_image = length-1


def make_grid(cols,rows):
  grid = [0]*cols
  for i in range(cols):
    with st.container():
      grid[i] = st.columns(rows)
  return grid


# layout of the page
grid = make_grid(2,1)

# datafile_chosen = {}
# datafile_chosen["spectrogram"] = True
# datafile_chosen["image"] = True

if datafile_chosen["spectrogram"]:
  grid[0][0].header(f"Spectrogram feature")

  col1_spectrogram,col2_spectrogram = grid[0][0].columns(2)

  if 'counter_spectrogram' not in st.session_state: 
    st.session_state.counter_spectrogram = 0

  # Get list of images in folder
  feature_spectrogram_subpath = r"spectrograms"
  feature_spectrogram_set = [os.path.join(feature_spectrogram_subpath,f) for f in os.listdir(feature_spectrogram_subpath)]
  col1_spectrogram.write(f"Total frames: {len(feature_spectrogram_set)}")
  col1_spectrogram.write(feature_spectrogram_set)

  forward_spec_btn = col1_spectrogram.button("Show next frame ⏭️",on_click=show_next_spectrogram,args=([len(feature_spectrogram_set)]), key="spectrogram_forward_btn")
  backward_spec_btn = col1_spectrogram.button("Show last frame ⏪",on_click=show_last_spectrogram,args=([len(feature_spectrogram_set)]), key="spectrogram_backward_btn")
  photo = feature_spectrogram_set[st.session_state.counter_spectrogram]
  col2_spectrogram.image(photo,caption=photo)
  col1_spectrogram.write(f"Index : {st.session_state.counter_spectrogram}")


# show camera images
if datafile_chosen["image"]:
  grid[1][0].header(f"Camera images")

  col1_image,col2_image = grid[1][0].columns(2)

  if 'counter_image' not in st.session_state: 
    st.session_state.counter_image = 0

  # Get list of images in folder
  params = {
    "file_path": datafile_chosen["path"],
    "file_name": datafile_chosen["name"],
    "config": datafile_chosen["config"],
  }
  response = requests.get(f"http://detection_vis_backend:8001/feature/{datafile_chosen['parse']}/image/0", params=params)
  feature = response.json()

  feature_subpath = r"images"
  feature_set = [os.path.join(feature_subpath,f) for f in os.listdir(feature_subpath)]
  col1_image.write(f"Total frames: {len(feature_set)}")
  col1_image.write(feature_set)
  forward_imag_btn = col1_image.button("Show next frame ⏭️",on_click=show_next_image,args=([len(feature_set)]), key="image_forward_btn")
  backward_imag_btn = col1_image.button("Show last frame ⏪",on_click=show_last_image,args=([len(feature_set)]), key="image_backward_btn")
  photo = feature_set[st.session_state.counter_image]
  col2_image.image(photo,caption=photo)
  col1_image.write(f"Index : {st.session_state.counter_image}")

