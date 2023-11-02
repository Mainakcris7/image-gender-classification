import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

st.title("Gender classifier")
img = st.file_uploader('Select an image')
if img:
    try:
        image = Image.open(img)
    except Exception as e:
        st.error('Unsupported file format !')
    try:
        image = np.array(image)
        image_tensor = tf.image.resize(image, size=(64, 64))
        st.image(image, width=224)
    except:
        st.error(e)


try:
    model = tf.keras.models.load_model('gender_detector.keras')
except Exception as e:
    st.write(e)

predict_btn = st.button('Predict')
if predict_btn:
    def pred_img(img):
        img = tf.expand_dims(img, axis=0)
        img = img/255.
        pred = False
        pred = model.predict(img)
        if pred[0][0] > 0.5:
            s = f"#### The image is predicted as a `MALE` with confidence `{pred[0][0]*100:.1f}%`"
            st.write(s)
        else:
            s = f"#### The image is predicted as a `FEMALE` with confidence `{(1 - pred[0][0]) *100:.1f}%`"
            st.write(s)

    try:
        pred_img(image_tensor)
    except:
        pass
