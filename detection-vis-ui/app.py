import requests

import streamlit as st
import pandas as pd

# Define the title 
st.title("Automotive sensor data visualization web application")

DATASETS = {
    "RaDICaL": "RaDICaL",
    "RADIal": "RADIal",
}

########################
response = requests.get("http://detection-vis-backend:8001/users")  # or your FastAPI server URL
data = response.json()
#df = pd.DataFrame(data)
st.json(data)
########################

dataset = st.selectbox("Choose the dataset", [i for i in DATASETS.keys()])
st.subheader(f"{dataset} Dataset")
st.write("RaDICaL dataset includes 4 subdatasets. Each subdataset contains a couple of `.bag` files which includes raw sensor recordings of RGB, depth and radar data streams.")

tab1, tab2, tab3 = st.tabs(["radar_high_res", "indoor_human", "30m_collection"])

with tab1:
  st.subheader("radar_high_res")
  # Show bag files 
  colms = st.columns((1, 2, 2, 1, 1))
  fields = ['No', 'File name', 'info', "size", 'action']
  for col, field_name in zip(colms, fields):
    # column header
    col.write(field_name)

  data_table = {'name':['file1', 'file2', 'file3'], 'info':['a', 'b', 'c'], 'size':[100,200,300]}
  for no, email in enumerate(data_table['name']):
    col1, col2, col3, col4, col5 = st.columns((1, 2, 2, 1, 1))
    col1.write(no)  # index
    col2.write(data_table['name'][no])  
    col3.write(data_table['info'][no])  
    col4.write(data_table['size'][no]) 
    
    button_type = "Load" 
    button_phold = col5.empty()  # create a placeholder
    do_action = button_phold.button(button_type, key=no)
    if do_action:
      bagfile = data_table['name'][no]
      # response = requests.post(f"http://backend:8080/load/{file}", file=bagfile)
      # datafile = response.json()
      button_phold.empty()  #  remove button
      col5.write('Loading...')


with tab2:
  st.subheader("indoor_human")
  #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
  st.subheader("30m_collection")
  #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


