
import streamlit as st
import requests
import os
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

from matplotlib.patches import Rectangle
from PIL import Image

from streamlit_ace import st_ace
from streamlit_extras.no_default_selectbox import selectbox


backend_service = os.getenv('BACKEND_SERVICE', 'localhost')
model_rootdir = os.getenv('MODEL_ROOTDIR')

st.set_page_config(
    page_title="Have fun ",
    page_icon=":red_car:", 
)



response = requests.get(f"http://{backend_service}:8001/models")  
models = response.json()

if 'model_chosen' in st.session_state:
  model_index = next((idx for idx, item in enumerate(models) if item['name'] == st.session_state.model_chosen), -1)
else:
  model_index = 0
model_chosen = st.selectbox("Choose a model", [item["name"] for item in models], index=model_index)
model_id = [item['id'] for item in models if item['name'] == model_chosen][0]

st.subheader(f"Model: {model_chosen}")
# st.write(model_id) # for debug
# show model lineage(datafiles, features, model config, train config, performance)

model_path = os.path.join(model_rootdir, model_chosen)

# in default choose the last epoch checkpoint model
files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)) and f.endswith('.pth')]
checkpoint_count = len(files)
checkpoint_id = st.selectbox("Choose a checkpoint(In default from the last epoch)", list(range(1,checkpoint_count+1)), index=checkpoint_count-1) - 1

# Get data&model&train info of the chosen model 
response = requests.get(f"http://{backend_service}:8001/model/{model_id}")  
model_paras = response.json()
tabs = st.tabs(["Data info", "Model info", "Train info", "Performance"])
for idx,tab in enumerate(tabs):
  with tab:
    if idx == 0:
      # Show data files
      st.write("Train data file(s):")
      s = ''
      for i in model_paras["datafiles"]:
        s += "- " + i["name"] + "\n"
      st.markdown(s)
      # Show features
      features_chosen = st.write("Train feature(s):")
      s = ''
      for i in model_paras["features"]:
        s += "- " + i + "\n"
      st.markdown(s)
    elif idx == 1 or idx == 2:
      cfg = ["", "model_config", "train_config"]
      formatted_configs = json.dumps(model_paras[cfg[idx]], indent=2)
      formatted_configs = st_ace(value=formatted_configs,theme="solarized_dark",
                keybinding="vscode",
                min_lines=20,
                max_lines=None,
                font_size=14,
                tab_size=4,
                wrap=False,
                show_gutter=True,
                show_print_margin=False,
                readonly=True,
                annotations=None)
    else:
      st.write("Val dataset")
      val_eval = pd.read_csv(os.path.join(model_path, "val_eval.csv"))
      st.table(val_eval)
      st.write("Test dataset")
      test_eval = pd.read_csv(os.path.join(model_path, "test_eval.csv"))
      st.table(test_eval)

# # Get sample split info of the chosen model
# with open(os.path.join(model_path, "samples_split.txt"), 'r') as f:
#   lines = f.readlines()
# sample_ids = {}
# for line in lines:
#   if "TRAIN_SAMPLE_IDS:" in line:
#     sample_ids["Train data"] = list(map(int, line.replace("TRAIN_SAMPLE_IDS: ", "").strip().split(',')))
#   elif "VAL_SAMPLE_IDS:" in line:
#     sample_ids["Val data"] = list(map(int, line.replace("VAL_SAMPLE_IDS: ", "").strip().split(',')))
#   elif "TEST_SAMPLE_IDS:" in line:
#     sample_ids["Test data"] = list(map(int, line.replace("TEST_SAMPLE_IDS: ", "").strip().split(',')))  


@st.cache_data
def predict(model_id, checkpoint_id, sample_id, file_id, split_type):
  params = {"checkpoint_id": checkpoint_id, "sample_id": sample_id, "file_id": file_id, "split_type": split_type}
  response = requests.get(f"http://{backend_service}:8001/predict/{model_id}", params=params)  
  res = response.json()
  return res["prediction"], res["feature_show_pred"]


@st.cache_data
def get_feature(file_id, feature, idx):
  response = requests.get(f"http://{backend_service}:8001/feature/{file_id}/{feature}/{idx}")
  feature_data = response.json()
  # serialized_feature = feature_data["serialized_feature"]
  # return serialized_feature
  return feature_data

@st.cache_data
def show_pred_with_gt(file_id, features, frame_id, pred_objs, feature_show_pred):
  feature = features[0]
  feature_data = get_feature(file_id, feature, frame_id)
  serialized_feature = feature_data["serialized_feature"]
  
  if feature == "lidarPC" or feature == "radarPC":
    feature_image = np.array(serialized_feature)
    plt.figure(figsize=(8, 6))
    plt.grid()
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"index: {frame_id}, shape: {feature_image.shape}", y=1.0)
    if feature_data["gt_label"]:
      objs = feature_data["gt_label"]
      if len(objs[0]) == 25:
        plt.scatter(feature_image[:, 0], feature_image[:, 1], c='darkblue', s=1, alpha=0.5)
        plt.xlim(0, 100) 
        plt.ylim(-50, 50)
        for obj in objs:
          points = obj[:24]
          points = np.array(points).reshape((8, 3))
          for k in range(0, 3):
            plt.plot(points[k:k + 2, 0], points[k:k + 2, 1], 'r-')
          plt.plot([points[3, 0], points[0, 0]], [points[3, 1], points[0, 1]], 'r-', linewidth=1)
          plt.text(points[0, 0] + 1, points[0, 1] + 1, obj[24], c='g')
      else:
        # Case: RADIal Dataset
        plt.plot(feature_image[:,1], feature_image[:,0], '.')
        plt.xlim(-20,20)
        plt.ylim(0,100)
        # plt.xlim(-max(abs(feature_image[:,1])), max(abs(feature_image[:,1])))
        # plt.ylim(0, max(feature_image[:,0])) 
        for obj in objs:
          plt.plot(obj[0],obj[1],'rs')
    else:
      # Case: RaDIcal Dataset
      plt.plot(feature_image[:,1], feature_image[:,0], '.')
      plt.xlim(-20,20)
      plt.ylim(0,50)
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    st.pyplot(plt)
  elif feature == "RD":
    feature_image = np.array(serialized_feature)
    plt.figure(figsize=(2, 3))
    ## Rotates RD image 90 degrees clockwise
    # feature_image = np.rot90(feature_image, k=-1) 
    # img_height, img_width = feature_image.shape
    plt.imshow(feature_image) #, aspect='auto'
    plt.xlabel('Doppler', fontsize=8)
    plt.ylabel('Range', fontsize=8)
    plt.title(f"index: {frame_id}, shape: {feature_image.shape}", y=1.0, fontsize=8)
    if feature_data["gt_label"]:
      objs = feature_data["gt_label"]
      if len(objs[0]) == 2: 
        # Case: RADIal Dataset
        for obj in objs:
          plt.plot(obj[1],obj[0],'ro', alpha=0.4) 
          # plt.plot(img_width - obj[0], obj[1], 'ro')  # Adjust coordinates for rotation
      elif len(objs[0]) == 5:
        # Case: CARRADA Dataset 
        for obj in objs:
          ## If RD image doesn't rotate
          rect = Rectangle((obj[0],obj[1]), obj[2]-obj[0], obj[3]-obj[1],linewidth=1, edgecolor='r', facecolor='none')
          plt.gca().add_patch(rect)
          plt.text(obj[0], obj[1] - 2, '%s' % obj[4], c='y', fontsize=6)
          # plt.yticks([0, 64, 128, 192, 255], [50, 37.5, 25, 12.5, 0]) # range
          # plt.xticks([0, 16, 32, 48, 63], [-13, -6.5, 0, 6.5, 13]) # doppler

          ## If RD image is rotated 90 degree clockwise:
          # rect = Rectangle((img_width - obj[0] - (obj[2] - obj[0]), obj[1]), obj[2] - obj[0], obj[3] - obj[1], linewidth=1, edgecolor='r', facecolor='none')
          # plt.gca().add_patch(rect)
          # plt.text(img_width - obj[0] - (obj[2] - obj[0]), obj[1] - 5, '%s' % obj[4], c='y')  

          ## If RD image is rotated 90 degree clockwise:
          # new_x = img_width - obj[3]
          # new_y = obj[0]
          # new_width = obj[3] - obj[1]
          # new_height = obj[2] - obj[0]
          # rect = Rectangle((new_x, new_y), new_width, new_height, linewidth=1, edgecolor='r', facecolor='none')
          # plt.gca().add_patch(rect)
          # plt.text(new_x, new_y - 2, '%s' % obj[4], c='y')
    st.pyplot(plt, use_container_width=False)
  elif feature == "RA":
    feature_image = np.array(serialized_feature)
    plt.figure(figsize=(8,6))
    plt.imshow(feature_image)
    plt.xlabel('Azimuth')
    plt.ylabel('Range')
    plt.title(f"index: {frame_id}, shape: {feature_image.shape}", y=1.0)
    if feature_data["gt_label"]:
      objs = feature_data["gt_label"]
      if len(objs[0]) == 3:
        # Case: CRUW Dataset
        for obj in objs:
          plt.plot(obj[1],obj[0],'ro', alpha=0.5)
          plt.text(obj[1] + 2, obj[0] + 2, f'gt:{obj[2]}', color='red')
      elif len(objs[0]) == 5:
        # Case: CARRADA Dataset
        for obj in objs:
          rect = Rectangle(np.array(obj[:2]), obj[2]-obj[0], obj[3]-obj[1],linewidth=1, edgecolor='r', facecolor='none')
          plt.gca().add_patch(rect)
          plt.text(obj[0], obj[1] -5, '%s' % obj[4], c='y')
    
    if feature_show_pred == "RA":
      classes = ["pedestrian", "cyclist", "car"]
      for obj in pred_objs:
        plt.plot(obj[1],obj[0],'yo', alpha=0.5)
        cls_id = int(obj[2])
        plt.text(obj[1] + 2, obj[0] + 2, f'pred:{classes[cls_id]}(conf:{obj[3]:.2f})', color='yellow')
    st.pyplot(plt)
  elif feature in ('image', 'depth_image'):
    plt.figure(figsize=(8,6))
    feature_image = mpimg.imread(serialized_feature)
    plt.imshow(feature_image) #, aspect='auto'
    plt.title(f"index: {frame_id}, shape: {feature_image.shape}", y=1.0)
    img_height, img_width, _ = feature_image.shape
    if feature_data["gt_label"]:
      objs = feature_data["gt_label"]
      if len(objs[0]) > 5:
        number_of_colors = len(objs)
        colorlist = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
        for n, obj in enumerate(objs):
          # Case: Astyx Dataset
          qs = obj[:16]
          qs = np.array(qs).reshape((2, 8))
          qs = qs.astype(np.int32)
          qs = np.transpose(qs)
          for k in range(0, 4):
            i, j = k, (k + 1) % 4
            plt.plot([qs[i, 0], qs[j, 0]], [qs[i, 1], qs[j, 1]], color=colorlist[n], linewidth=1)
            i, j = k + 4, (k + 1) % 4 + 4
            plt.plot([qs[i, 0], qs[j, 0]], [qs[i, 1], qs[j, 1]], color=colorlist[n], linewidth=1)
            i, j = k, k + 4
            plt.plot([qs[i, 0], qs[j, 0]], [qs[i, 1], qs[j, 1]], color=colorlist[n], linewidth=1) 
          #plt.text(qs[0, 0], qs[1, 0], obj[16], c=colorlist[n])
      else:
        for obj in objs:
          # Case: RADIal Dataset
          rect = Rectangle(np.array(obj[:2]), obj[2]-obj[0], obj[3]-obj[1],linewidth=1, edgecolor='r', facecolor='none')
          plt.gca().add_patch(rect)
    plt.xlim(0, img_width - 1)
    plt.ylim(img_height - 1, 0)
    st.pyplot(plt)
  else:
    feature_image = feature_image - np.min(feature_image)
    feature_image = feature_image / np.max(feature_image)
    st.image(feature_image, caption=f"index: {frame_id}, shape: {feature_image.shape}")
  
  if feature != "image" and feature_show_pred != "image":
    feature_data = get_feature(file_id, "image", frame_id)
    image_path = feature_data["serialized_feature"]
    image = mpimg.imread(image_path)
    img_height, img_width, _ = image.shape
    plt.figure(figsize=(8,6))
    plt.imshow(image) #, aspect='auto'
    plt.title(f"index: {frame_id}, shape: {feature_image.shape}", y=1.0)
    plt.xlim(0, img_width - 1)
    plt.ylim(img_height - 1, 0)
    st.pyplot(plt)

  return

radio_options = ["Train data", "Val data", "Test data", "I want to upload my own data file to do inference"]
radio_select = st.radio("Which data you want to check the detection model from?", radio_options)

if radio_select is radio_options[-1]:
  uploaded_file = st.file_uploader("Choose a data file", accept_multiple_files=False)
  if uploaded_file is not None:
    #bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    im = Image.open(uploaded_file)
    im = im.resize((600, 400))
    image = np.array(im)
    st.image(image)
else:
  radio_dict = {"Train data": "train", "Val data": "val", "Test data": "test"}
  split_type = radio_dict[radio_select]
  frame_begin = 0
  file_path = os.path.join(model_path, f"{split_type}_sample_ids.csv")
  samples_info = pd.read_csv(file_path, index_col=0).to_numpy()
  frame_end = len(samples_info)
  frame_id = st.slider('Choose a data frame', frame_begin, frame_end, frame_begin)
  if samples_info.shape[1] == 1:
    original_sample_id = samples_info[frame_id, 0]
    st.write(f"Original from: No. {original_sample_id} sample of the whole dataset")
    file_id = model_paras["datafiles"][0]["id"]
  else:
    file_name, file_id, original_sample_id = samples_info[frame_id, 1], samples_info[frame_id, 2], samples_info[frame_id, 3]
    st.write(f"Original from: No. {original_sample_id} sample of datafile {file_name}")
  # Check the parsing state of the data file where the chosen sample is
  original_datafile_parsed = f"parse_{file_id}"
  if original_datafile_parsed not in st.session_state:
    st.session_state[original_datafile_parsed] = False
    response = requests.get(f"http://{backend_service}:8001/parse/{file_id}")
    if response.status_code != 204:
      st.info(f"An error occurred in parsing data file {file_name}.")
      st.stop()
    else:
      st.session_state[original_datafile_parsed] = True 
  pred_objs, feature_show_pred = predict(model_id, checkpoint_id, original_sample_id, file_id, split_type)
  # Get gt label info and show together with prediction
  # feature_data = get_feature(file_id, feature, original_sample_id)
  # serialized_feature = feature_data["serialized_feature"]
  # feature_image = np.array(serialized_feature)
  # st.image(pred_image, caption="Prediction 🟦 Ground Truth 🟥")
  show_pred_with_gt(file_id, model_paras["features"], original_sample_id, pred_objs, feature_show_pred)
  # pred_image = np.array(res)
  # st.image(pred_image, caption="Prediction 🟦 Ground Truth 🟥")