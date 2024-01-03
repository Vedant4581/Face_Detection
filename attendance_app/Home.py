import streamlit as st
import pandas as pd
import cv2


st.set_page_config(page_title="Attendance System", layout="wide")

st.header('Attendance System Using Face Recognition')

with st.spinner("Loading Models and connecting to redis db..."):
    import face_recognition

st.success('Model loaded successfully')
st.success('Redis db successfully connected')