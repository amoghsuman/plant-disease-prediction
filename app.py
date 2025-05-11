import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the model once
@st.cache_resource
def load_model_cached():
    return load_model('plant_disease.h5')

model = load_model_cached()

# Class names
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

st.title("🌿 Plant Disease Detection")
st.markdown("Upload a leaf image to predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Immediately read and process the image (do not store across reruns)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Resize and preprocess
        image = cv2.resize(image, (256, 256))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict
        prediction = model.predict(image)
        result = CLASS_NAMES[np.argmax(prediction)]

        crop, disease = result.split('-')
        st.success(f"🟢 This is a **{crop}** leaf with **{disease.replace('_', ' ')}**.")
    else:
        st.error("Failed to process the image. Please upload a valid JPG/PNG file.")
