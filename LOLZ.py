import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

#####################################################################################################################
# DEPLOY MODEL!!!!!!

#Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#function to detect
def detect_objects(image):
    image = Image.fromarray(image).convert('RGB')
    results = model(image)
    return results
#####################################################################################################################

#####################################################################################################################
#OUTLINE OF WEBSITE !!!!!!!!

#INDIVIDUAL TABS
def infoTAB():
    st.write("Learn More")
    
def demoTAB():
    st.title("Trashy classifier")
    option = st.radio("Choose Input Method:", ('Upload Image', 'Use Camera'))
    if option == 'Upload Image':
        uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            image = np.array(Image.open(uploaded_image))
            results = detect_objects(image)
            st.image(np.array(results.render()))  
    else:
        st.title("Webcam Live Feed")
        run = st.button('Run')
        end = st.button('End Feed')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to retrieve frame from webcam.")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            results = detect_objects(frame)
            FRAME_WINDOW.image(np.array(results.render()))
        camera.release()
        cv2.destroyAllWindows()
        if end:
            st.write('Webcam Stopped')
            camera.release()
            cv2.destroyAllWindows()

#####################################################################################################################
#  RUN STREAMLIT !!!!!!!!!
def main():
    tab_select = st.sidebar.radio("Welcome! Where would you like to explore?", ("Know More About Our Model!", "Try Out Our Model!"))
    if tab_select == "Know More About Our Model!":
        infoTAB()
    elif tab_select == "Try Out Our Model!":
        demoTAB()
if __name__ == '__main__':
    main()
#####################################################################################################################



