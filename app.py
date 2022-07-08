#Code - @ParthVyas
import streamlit as st #importing streamlit and tensorflow
import tensorflow as tf
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
from PIL import Image ,ImageOps


st.set_option('deprecation.showfileUploaderEncoding',False) 

@st.cache(allow_output_mutation=True) 

def load_model(): #loading our model
  model = tf.keras.models.load_model(r'Model\BrainTumorModel .h5')
  return model

model = load_model()
#defining the header or title of the page that the user will be seeing

st.markdown("<h1 style='text-align: center; color: Black;'>Brain Tumor Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: Black;'>All you have to do is Upload the MRI scan and the model will do the rest!</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: Black;'>Final Year Project - MSRIT</h4>", unsafe_allow_html=True)

file=st.file_uploader("Please upload your MRI Scan",type = ["jpg","png"]) 

def import_and_predict(image_data,model): 
  size = (150,150)
  image1 = ImageOps.fit(image_data,size,Image.ANTIALIAS)
  image = ImageOps.grayscale(image1)
  img = np.asarray(image)
  img_reshape = img[np.newaxis,...]
  #img_reshape = img_reshape/255.0
  img_reshape = img.reshape(1,150,150,1)
  prediction = model.predict(img_reshape)
  return prediction

if file is None:
  st.markdown("<h5 style='text-align: center; color: Black;'>Please Upload a File</h5>", unsafe_allow_html=True)
else:
  image = Image.open(file)
  st.image(image,use_column_width = True)
  predictions = import_and_predict(image,model)
  class_names = ['Tumor','Tumor','No Tumor','Tumor']
  string = "The patient most likely has:"+ class_names[np.argmax(predictions)]
  st.success(string)
  