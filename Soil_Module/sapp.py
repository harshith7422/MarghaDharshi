import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore

# Load Trained Model
MODEL_PATH = "./models/soil_classifier_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define Image Size
IMG_SIZE = (128, 128)

# Define Soil Classes
SOIL_CLASSES = ['Alluvial', 'Black', 'Laterite', 'Red']

# Streamlit UI Configuration
st.set_page_config(page_title="Soil Classification", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        body { font-family: 'Inter', sans-serif; }
        .stButton>button { border-radius: 8px; padding: 10px 20px; font-size: 16px; }
        .stTextInput>div>div>input { border-radius: 8px; }
        .stFileUploader>div>button { background-color: #3A3B3C; color: white; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# UI Layout
st.title("🌱 MarghaDharshi - Soil Classification")
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a soil image", type=["jpg", "png", "jpeg"], label_visibility='hidden')
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True, output_format='auto')

with col2:
    if uploaded_file:
        # Process Image
        img = image.load_img(uploaded_file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict Soil Type
        prediction = model.predict(img_array)
        predicted_class = SOIL_CLASSES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display Results
        st.subheader("🧪 Prediction Results")
        st.write(f"**Soil Type:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        st.success("Prediction Completed ✅")
