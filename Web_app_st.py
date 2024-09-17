import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import keras
from keras.preprocessing import image
from keras.utils import custom_object_scope
from keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input 

model = tf.keras.models.load_model("model_cnn.h5")
model_vgg = tf.keras.models.load_model("model_vgg16.h5")
model_res = tf.keras.models.load_model("model_resnet50.h5")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.theculturetrip.com/wp-content/uploads/2018/05/shutterstock_1097144396.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Detecting Fake Currency using Deep Learning Technique CNN</p>'
st.markdown(new_title, unsafe_allow_html=True)
### load file

st.markdown("<span style='font-size:25px;color:white;'>We Designed our project using CNN, Resnet50 and VGG16 models For the detection of Fake and Real currency.</span>",unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a image file", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file)
    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        img = tf.keras.utils.load_img(uploaded_file,target_size=(120, 120, 3))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])

        res_cnn = model.predict(input_arr)
        res_vgg    =model_vgg.predict(input_arr)
        res_resnet = model_res.predict(input_arr)
        #plt.imshow(img)
        

        resf = res_cnn[0][0]
        resr = res_cnn[0][1]
        
        fv=res_vgg[0][0]
        rv=res_vgg[0][1]
        
        fr=res_resnet[0][0]
        rr=res_resnet[0][1]

        resf1=(fr+fv+resf)/3
        resr1=(rr+rv+resr)/3

        if resr1>=0.5:st.markdown("<span style='font-size:30px;color:white;'>The given currency is Real !!! </span>",unsafe_allow_html=True)
            
        else:
            st.markdown("<span style='font-size:30px;color:white;'>The given currency is Fake !!! </span>",unsafe_allow_html=True)

        st.markdown("<span style='font-size:30px;color:white;'>The given currency is {} Real ! </span>".format(resr1),unsafe_allow_html=True)
        st.markdown("<span style='font-size:30px;color:white;'>The given currency is {} Fake ! </span>".format(resf1),unsafe_allow_html=True)

