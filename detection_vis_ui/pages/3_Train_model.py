
import streamlit as st
import requests
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from streamlit_extras.switch_page_button import switch_page



backend_service = os.getenv('BACKEND_SERVICE', 'localhost')

st.set_page_config(
    page_title = "Have fun ",
    page_icon = ":red_car:", 
    layout = "wide"
)


if 'datafiles_chosen' not in st.session_state or 'features_chosen' not in st.session_state:
  st.info("Please choose data and feature first.")
  st.stop()
