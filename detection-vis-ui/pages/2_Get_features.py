import streamlit as st


st.set_page_config(
    page_title="Have fun ",
    page_icon=":red_car:", 
)


feature_name = st.selectbox("Get the feature", [i["name"] for i in root_datasets])
