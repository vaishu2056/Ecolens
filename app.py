import streamlit as st

st.set_page_config(layout="wide")  # optional

st.title("Land Cover Classification using Deep Learning")
st.write("Upload an image to classify land types:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    import cv2
    import numpy as np
    from PIL import Image

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (resize, normalize, etc.)
    # Assuming IMAGE_SIZE is predefined:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict with your model
    prediction = model.predict(image_array)

    # Show output (e.g., segmentation map or label)
    st.write("Prediction result:")
    # You can replace this with actual label map or image mask
    st.write(prediction)
