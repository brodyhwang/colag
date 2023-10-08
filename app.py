import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('keras_model.h5')

with open('labels.txt', 'r') as file:
    class_labels = file.read().splitlines()

st.title("Guitar Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    confidence = np.max(prediction) * 100

    st.write(f"Predicted Guitar Type: {predicted_label}")
    st.write(f"Confidence: {confidence:.2f}%")