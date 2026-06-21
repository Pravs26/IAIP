import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import img_to_array

st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #D63384;
}
.sub-title {
    text-align: center;
    font-size: 20px;
    color: #555;
}
.result-box {
    padding: 18px;
    border-radius: 15px;
    background-color: #E8F8F0;
    color: #0B6B3A;
    font-size: 17px;
    font-weight: 600;
    text-align: center;
    min-height: 150px;
}
.note-box {
    padding: 12px;
    border-radius: 10px;
    background-color: #FFF4D6;
    color: #6B4E00;
    font-size: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🌸 Real-Time Flower Image Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload multiple flower images and let AI classify them instantly.</div>',
    unsafe_allow_html=True
)

st.write("---")

model = tf.keras.models.load_model("saved_model/flower_classifier.keras")

class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

flower_info = {
    "daisy": "Daisy represents purity and innocence, commonly seen with white petals and a yellow center.",
    "dandelion": "Dandelion is known for its bright yellow flower and fluffy seed head.",
    "roses": "Roses are popular ornamental flowers often associated with love and beauty.",
    "sunflowers": "Sunflowers are bright yellow flowers that naturally face toward sunlight.",
    "tulips": "Tulips are colorful spring flowers known for their beautiful cup-shaped petals."
}

img_height = 224
img_width = 224

uploaded_files = st.file_uploader(
    "Upload flower images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write("## Prediction Results")

    cols = st.columns(3)

    for index, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")

        image_resized = image.resize((img_width, img_height))
        img_array = img_to_array(image_resized)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        probabilities = score.numpy() * 100

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        flower_description = flower_info[predicted_class]

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_classes = [class_names[i] for i in top_indices]
        top_probs = [probabilities[i] for i in top_indices]

        prob_df = pd.DataFrame({
            "Flower Type": top_classes,
            "Probability (%)": top_probs
        })

        with cols[index % 3]:
            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

            st.markdown(
                f"""
                <div class="result-box">
                    🌼 Prediction: {predicted_class}<br>
                    ✅ Confidence: {confidence:.2f}%<br><br>
                    📝 {flower_description}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(int(confidence))
            st.caption(f"Confidence Score: {confidence:.2f}%")

            st.write("📊 Top 3 Prediction Confidence")
            st.bar_chart(
                prob_df,
                x="Flower Type",
                y="Probability (%)"
            )

            st.markdown(
                """
                <div class="note-box">
                    Higher confidence means the model strongly matched this flower category.
                </div>
                """,
                unsafe_allow_html=True
            )

else:
    st.info("Please upload one or more flower images to start classification.")