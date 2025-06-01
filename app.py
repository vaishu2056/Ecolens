import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load model safely
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/segmentation_model.h5")  # Make sure this path exists!
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

if model is None:
    st.stop()  # Don't proceed if model fails to load
# Constants
IMAGE_SIZE = 256
NUM_CLASSES = 5  # Update based on your model

# Streamlit UI
st.set_page_config(layout="centered", page_title="Land Cover Classification")
st.title("Land Cover Classification using Deep Learning")
st.write("Upload an image to classify land types:")

# Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred_mask = model.predict(img_array)[0]  # Shape: (256, 256, NUM_CLASSES)
    pred_mask = np.argmax(pred_mask, axis=-1)  # Shape: (256, 256)

    # Optional: apply a color map
    def apply_colormap(mask):
        colormap = np.array([
            [0, 0, 0],         # Class 0 - Black
            [0, 255, 0],       # Class 1 - Green
            [0, 0, 255],       # Class 2 - Blue
            [255, 255, 0],     # Class 3 - Yellow
            [255, 0, 0]        # Class 4 - Red
        ])
        return colormap[mask]

    colored_mask = apply_colormap(pred_mask.astype(np.uint8))

    # Display result
    st.subheader("Predicted Segmentation Mask")
    st.image(colored_mask, clamp=True, use_column_width=True)
