import tensorflow as tf
import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="LeafLens - Plant Disease Detection",
    layout="wide",
    page_icon="🌿"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f4fff6;
}

/* Header */
.header {
    background-color: #4caf50;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
}

/* Footer */
.footer {
    background-color: #2e7d32;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-top: 30px;
}

/* Prediction Card */
.result-card {
    background-color: #e8f5e9;
    padding: 20px;
    border-radius: 15px;
    border-left: 8px solid #2e7d32;
    font-size: 18px;
}

/* Remedy Box */
.remedy-box {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border: 2px dashed #66bb6a;
    margin-top: 10px;
}

/* Section Title */
.section-title {
    color: #1b5e20;
    font-weight: bold;
    font-size: 22px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model.h5",
        compile=False,
        safe_mode=False   
    )

model = load_model()

with open("class_names.json") as f:
    class_names = json.load(f)

# ---------------- FUNCTIONS ----------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def clean_label(label):
    return label.replace("___", " → ").replace("_", " ")

# ---------------- REMEDIES ----------------
remedies = {
    "healthy": "✅ No disease detected. Maintain proper watering, sunlight, and soil health.",

    "Powdery mildew": "🌿 Use sulfur-based fungicide. Improve air circulation.",
    "Leaf mold": "🍃 Avoid overhead watering. Use copper fungicide.",
    "Bacterial spot": "🧴 Remove infected leaves and use bactericides.",
    "Early blight": "🌱 Practice crop rotation and apply fungicide.",
    "Late blight": "⚠️ Remove infected plants immediately. Use strong fungicides.",
    "Rust": "🌾 Apply neem oil or sulfur spray regularly.",
}

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
show_chart = st.sidebar.checkbox("Show Confidence Chart", True)
show_top = st.sidebar.checkbox("Show Top Predictions", True)

st.sidebar.info("Upload a plant leaf image to detect disease.")

# ---------------- HEADER ----------------

st.markdown(
    """
    <div class="header">
        <div style="font-size:36px; font-weight:bold;">LeafLens 🔍</div>
        <div style="font-size:22px;">🌿 Plant Leaf Disease Detection System</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    # ---------------- IMAGE ----------------
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)

    # ---------------- PREDICTION ----------------
    img = preprocess_image(image)
    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    clean_name = clean_label(predicted_class)

    # ---------------- RESULT ----------------
    with col2:
        st.subheader("🧠 Prediction Result")

        st.markdown(f"""
        <div class="result-card">
            🌱 <b>Prediction:</b> {clean_name} <br>
            📊 <b>Confidence:</b> {confidence:.2f}
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence * 100))

        if confidence > 0.9:
            st.success("High confidence prediction ✅")
        elif confidence > 0.7:
            st.warning("Moderate confidence ⚠️")
        else:
            st.error("Low confidence ❌")

    st.markdown("---")

if uploaded_file:

    image = Image.open(uploaded_file)

    img = preprocess_image(image)
    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    clean_name = clean_label(predicted_class)

    # ---------------- RESULT ----------------
    st.markdown(f"Prediction: {clean_name}")

    # ---------------- REMEDY (FIXED) ----------------
    st.markdown('<p class="section-title">💊 Suggested Remedy</p>', unsafe_allow_html=True)

    if "healthy" in predicted_class.lower():

        st.markdown("""
        <div class="remedy-box">
        ✅ No disease detected. The plant is healthy.
        </div>
        """, unsafe_allow_html=True)

    else:
        remedy_text = "⚠️ General care: Remove infected leaves and use fungicide."

        for key in remedies.keys():
            if key.lower() in clean_name.lower():
                remedy_text = remedies[key]
                break

        st.markdown(f"""
        <div class="remedy-box">
        {remedy_text}
        </div>
        """, unsafe_allow_html=True)

    # ---------------- TOP PREDICTIONS ----------------
    if show_top:
        st.subheader("🔝 Top 5 Predictions")

        top5_idx = np.argsort(prediction[0])[-5:][::-1]

        for i in top5_idx:
            st.write(f"{clean_label(class_names[i])} : {prediction[0][i]:.4f}")

    # ---------------- CHART ----------------
    if show_chart:
        st.subheader("📊 Confidence Distribution")

        fig, ax = plt.subplots()
        ax.barh(
            [clean_label(c) for c in class_names],
            prediction[0]
        )
        ax.set_xlabel("Probability")
        st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
            <h5><b>&copy; LeafLens 2026 , Made by: Vansh Nagpal</b></h5>
🌿 Built with Streamlit and Deep Learning | AI Plant Disease Detection System
</div>
""", unsafe_allow_html=True)
