import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array

# Load model
model = tf.keras.models.load_model(
    "saved_model/flower_classifier.keras"
)

# Flower classes
class_names = [
    'daisy',
    'dandelion',
    'roses',
    'sunflowers',
    'tulips'
]

# Image settings
img_height = 224
img_width = 224

# Title
st.title(" Real-Time Flower Classifier")

st.write(
    "Upload a flower image and the AI model "
    "will predict the flower category."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Flower Image",
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize(
        (img_width, img_height)
    )

    # Convert image to array
    img_array = img_to_array(image_resized)

    # Expand dimensions
    img_array = tf.expand_dims(img_array, 0)

    # Prediction
    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]

    confidence = 100 * np.max(score)

    # Output
    st.success(
        f"This flower is most likely "
        f"**{predicted_class}** "
        f"with **{confidence:.2f}%** confidence."
    )