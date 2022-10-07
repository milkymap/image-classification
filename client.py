import cv2
import pickle, json 

import requests 

import zmq 
import numpy as np 
import itertools as it, functools as ft 

import streamlit as st
import pandas as pd 

from PIL import Image 

blank = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 255)

st.set_page_config(page_title="dog cat classification", layout="centered")

st.markdown("""
<style>
h1 { color: #111; font-family: 'Helvetica Neue', sans-serif; font-size: 30px; font-weight: bold; letter-spacing: -1px; line-height: 1; text-align: center; }
.big-font {
    font-weight:bold;
    font-family:Cursive;
    font-size:25px !important;
}
.button0 {
  font-weight:bold;
  font-family:Cursive;
  width: 100%;
  background-color: #4CAF50;
  border: none;
  color: white;
  padding: 15px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 25px;
  margin: 4px 2px;
  cursor: pointer;
  border-radius: 12px;
}
.button1 {background-color: #008CBA;} /* Blue */
.button2 {background-color: #f44336;} /* Red */ 
.button3 {background-color: #e7e7e7; color: black;} /* Gray */ 
.button4 {background-color: #555555;} /* Black */
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center'>Automatic Image Captioning</h1>", unsafe_allow_html=True)

st.sidebar.write(
    f"dog cat classification"
)

lunch = None 
binary = None 
bgr_image = None 
response = None

file_data = st.file_uploader(label='choose your image', type=['jpg', 'jpeg'])
beam_width = st.slider('beam width', 3, 11)
col1, col2 = st.columns(2)
with col1:
    if file_data is not None:
        binary = np.frombuffer(file_data.read(), np.uint8)
        bgr_image = cv2.imdecode(binary, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image).resize((512, 512))
        st.image(pil_image)
    else:
        st.image(blank)


lunch = st.markdown('<button class="button0">make prediction</button>', unsafe_allow_html=True)

with col2:
    if lunch is not None:
        if bgr_image is not None:
            response = requests.post('http://localhost:5000/predict', files={'data': binary})
            content = response.content
            print(content)
            msg = f'<p class="big-font">{content}</p>'
            st.markdown(msg, unsafe_allow_html=True)
    else:
        st.image(blank)
