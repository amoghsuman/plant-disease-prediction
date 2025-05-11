import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from io import BytesIO

# Load the model once
@st.cache_resource
def load_model_cached():
    return load_model("plant_disease.h5")

model = load_model_cached()

# Class labels
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# App UI
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of the plant leaf to predict the disease type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if file is uploaded
if uploaded_file is not None:
    try:
        # Read file as bytes (do not store in session_state!)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("⚠️ Unable to decode image. Please upload a valid file.")
        else:
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            image = cv2.resize(image, (256, 256))
            image = image.astype("float32") / 255.0
            image = np.expand_dims(image, axis=0)

            # Predict
            preds = model.predict(image)
            label = CLASS_NAMES[np.argmax(preds)]
            crop, disease = label.split("-")
            st.success(f"🟢 Prediction: **{crop}** leaf with **{disease.replace('_', ' ')}**")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
