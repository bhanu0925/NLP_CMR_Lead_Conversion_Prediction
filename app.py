from src.predict import get_prediction
from  src import config
import numpy as np
from src.datapreprocessing import data_cleaning
from src.utils import load_model

import warnings
from sklearn.exceptions import DataConversionWarning
 
import streamlit as st
#warnings.filterwarnings(action='ignore', category=DataConversionWarning)

st.set_page_config(page_title="CMR Lead Conversion Prediction",             
                layout="wide")

st.markdown("<h1 align='center'>CMR Lead Conversion Prediction</h1>", unsafe_allow_html=True)
st.image("https://images.news18.com/ibnlive/uploads/2021/11/road-accident-163747048616x9.jpg", use_column_width=True)
#https://cdn.technologyadvice.com/wp-content/uploads/2020/02/Optimize-conversion-rate-01.png
st.sidebar.title("About this application")
st.sidebar.write("""
        The app is aimed at predicting the lead conversion rate of a edtech company.
        """)

st.sidebar.info("### Made by:    Bhanumathi Ramesh")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/bhanumathiramesh)")
st.sidebar.markdown("[Github](https://github.com/bhanu0925/NLP_CMR_Text_Classification)")

st.markdown("<h2 align='center'> CMR Lead Conversion Prediction </h2>", unsafe_allow_html=True)


text = st.text_area('Enter the text to predict the lead conversion rate', '', height=200)
location = st.selectbox("Location ", options = config.locations )

if st.button("Predict") :
    
    if text != "":
        text = data_cleaning(text)
        model = load_model(model_path =r'model/svc_tfidfGS.pickle' )
        pred = get_prediction(text,location ,model)
        st.success(pred)
    else:
        st.error("Please enter the text")   
