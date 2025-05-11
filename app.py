# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tempfile

# Load the model (ensure plant_disease.h5 is in the same folder as app.py)
model = load_model('plant_disease.h5')

# Class names
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Streamlit app UI
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of the plant leaf to predict the disease type.")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict button
if st.button('Predict'):
    if plant_image is not None:
        # Read image file
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the image
        st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image")
        #st.write(f"Original image shape: {opencv_image.shape}")

        # Resize and reshape for model input
        opencv_image = cv2.resize(opencv_image, (256, 256))
        opencv_image = opencv_image.astype('float32') / 255.0  # normalize
        opencv_image = np.expand_dims(opencv_image, axis=0)    # shape: (1, 256, 256, 3)

        # Make prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]

        # Display result
        crop, disease = result.split('-')
        st.success(f"🟢 Prediction: This is a **{crop}** leaf with **{disease.replace('_', ' ')}**.")
    else:
        st.error("❌ Please upload an image before clicking Predict.")
