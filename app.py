import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the model once at startup
@st.cache_resource
def load_trained_model():
    return load_model('plant_disease.h5')

model = load_trained_model()

# Class labels
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of a plant leaf to detect possible diseases.")

# Upload the image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict if both image is uploaded and button is clicked
if plant_image is not None:
    # Convert the image to OpenCV format
    file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image", use_column_width=True)

    # Resize and preprocess
    opencv_image = cv2.resize(opencv_image, (256, 256))
    opencv_image = opencv_image.astype('float32') / 255.0
    opencv_image = np.expand_dims(opencv_image, axis=0)

    # Prediction
    Y_pred = model.predict(opencv_image)
    result = CLASS_NAMES[np.argmax(Y_pred)]

    crop, disease = result.split('-')
    st.success(f"🟢 Prediction: This is a **{crop}** leaf with **{disease.replace('_', ' ')}**.")
