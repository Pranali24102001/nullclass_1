#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import streamlit as st
from PIL import Image
import numpy as np


# Function to visualize activation map (eye regions)
def visualize_activation_map(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply face detection to get face coordinates
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw rectangles around detected faces
    for (x,y,w,h) in faces:
        # Highlight eye regions (for demonstration)
        eye_region1 = (x + w//4, y + h//4, w//4, h//4)
        eye_region2 = (x + w//2, y + h//4, w//4, h//4)
        cv2.rectangle(image, (eye_region1[0], eye_region1[1]), (eye_region1[0] + eye_region1[2], eye_region1[1] + eye_region1[3]), (255,0,0), 2)
        cv2.rectangle(image, (eye_region2[0], eye_region2[1]), (eye_region2[0] + eye_region2[2], eye_region2[1] + eye_region2[3]), (255,0,0), 2)
    return image

# Streamlit setup
st.title("Activation Map Visualization for Age Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting Faces and Visualizing Activation Map...")

    # Estimate age and visualize activation map
    activation_map_image = visualize_activation_map(image_np)

    st.image(activation_map_image,use_column_width=True)


# In[ ]:




