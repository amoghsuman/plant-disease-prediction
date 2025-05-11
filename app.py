# Library imports
import numpy as np
import streamlit as st
import cv2
import os
from keras.models import load_model

# Load the model (cached for performance)
@st.cache_resource
def load_model_cached():
    return load_model("plant_disease.h5")

model = load_model_cached()

# Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# App title
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of a plant leaf to detect possible diseases.")

# Upload image (no 'type=' to avoid StreamlitAPIException)
plant_image = st.file_uploader("Upload a JPG or PNG image")

# Predict button
submit = st.button("Predict")

# On predict click
if submit:
    if plant_image is None:
        st.error("⚠️ Please upload an image before clicking Predict.")
    else:
        # Validate extension manually (case-insensitive)
        ext = os.path.splitext(plant_image.name)[-1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            st.error(f"❌ Invalid file extension: `{ext}`. Please upload a JPG or PNG file.")
        else:
            # Read and decode image
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            if opencv_image is None:
                st.error("⚠️ Unable to process image. Please upload a valid image file.")
            else:
                # Show image
                st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image", use_container_width=True)

                # Preprocess for model
                opencv_image = cv2.resize(opencv_image, (256, 256))
                opencv_image = opencv_image.astype("float32") / 255.0
                opencv_image = np.expand_dims(opencv_image, axis=0)

                # Predict
                prediction = model.predict(opencv_image)
                predicted_label = CLASS_NAMES[np.argmax(prediction)]

                # Parse label
                crop, disease = predicted_label.split("-")
                confidence = round(100 * np.max(prediction), 2)

                # Display result
                st.success(f"🟢 This is a **{crop}** leaf with **{disease.replace('_', ' ')}**.")
                st.info(f"🤖 Model Confidence: **{confidence}%**")
