import streamlit as st


st.set_page_config(
    page_title="Have fun ",
    page_icon=":red_car:", 
)

# Define the title 
st.title("Automotive sensor data visualization web application")

st.write("This project demonstrates the following features into an interactive Streamlit app.")

st.write("1. Visualize data & features of open source automotive datasets.")
st.write("2. Train object detection model with selected data & feature & model & train configs from supported model types and persist the model.")
st.write("3. Object detection inference with persisted models.")

st.write("👈 Please select Choose data in the sidebar to start if you want 1 or 2.")
st.write("👈 Please select Inference in the sidebar to start if you want 3.")

  
