import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load model (cached to avoid reloading on every rerun)
@st.cache_resource
def load_model_cached():
    return load_model("plant_disease.h5")

model = load_model_cached()

# Class labels
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Streamlit UI
st.title("🌿 Plant Disease Identification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

        # Preprocess for model
        img_resized = img.resize((256, 256))
        img_array = np.array(img_resized) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

        # Make prediction
        prediction = model.predict(img_array)
        result = CLASS_NAMES[np.argmax(prediction)]

        crop, disease = result.split("-")
        st.success(f"🟢 Prediction: **{crop}** leaf with **{disease.replace('_', ' ')}**")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
