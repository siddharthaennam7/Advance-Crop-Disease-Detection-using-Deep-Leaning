import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Crop Disease Detection", layout="wide", page_icon="🌿")

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e8f5e9, #f1f8e9);
    font-family: 'Segoe UI', sans-serif;
}
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #1b5e20;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 10px 20px rgba(0,0,0,0.1);
}
.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.healthy {
    background-color: #dcedc8;
    color: #1b5e20;
}
.diseased {
    background-color: #ffcdd2;
    color: #b71c1c;
}
.remedy {
    background-color: #fff3cd;
    color: #6d4c41;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    font-weight: 500;
}
.model-box {
    background-color: #e3f2fd;
    color: #0d47a1;
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🌿 AI-Powered Crop Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Farming using Deep Learning</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model('model/crop_model.h5')
IMG_SIZE = 128

# 👉 Update if needed
class_names = ["Healthy", "Leaf Spot", "Blight", "Rust"]

# ---------------- REMEDIES ----------------
remedies_dict = {
    "Leaf Spot": [
        "Remove infected leaves",
        "Use copper fungicide spray",
        "Avoid overhead watering",
        "Ensure airflow"
    ],
    "Blight": [
        "Cut affected areas",
        "Use neem oil spray",
        "Improve drainage",
        "Avoid excess fertilizer"
    ],
    "Rust": [
        "Remove infected leaves",
        "Use sulfur fungicide",
        "Keep leaves dry",
        "Use resistant plants"
    ],
    "Diseased": [
        "Remove infected leaves",
        "Use fungicide spray",
        "Avoid overwatering",
        "Provide sunlight and airflow"
    ]
}

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# -------- LEFT --------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📸 Upload Leaf Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        predict_btn = st.button("🔍 Predict")
    else:
        st.info("Upload an image to begin")
        predict_btn = False

    st.markdown('</div>', unsafe_allow_html=True)

# -------- RIGHT --------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Prediction Result")

    if uploaded_file and predict_btn:

        img = image.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        # 🔥 AUTO DETECT MODEL TYPE
        if prediction.shape[1] == 1:
            pred_val = float(prediction[0][0])

            # 🔥 FIXED REVERSED LOGIC
            if pred_val > 0.5:
                disease = "Healthy"
                confidence = pred_val * 100
            else:
                disease = "Diseased"
                confidence = (1 - pred_val) * 100

        else:
            pred_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100
            disease = class_names[pred_index]

        # -------- RESULT --------
        if disease == "Healthy":
            st.markdown(
                f'<div class="result healthy">✅ Healthy Leaf<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result diseased">🌿 Disease: {disease}<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

            # 🌿 Remedies Section (SOFT COLOR)
            st.markdown("<h4 style='color:#6d4c41;'>🌿 Suggested Remedies:</h4>", unsafe_allow_html=True)

            for r in remedies_dict.get(disease, remedies_dict["Diseased"]):
                st.markdown(f"<p style='color:#6d4c41;'>• {r}</p>", unsafe_allow_html=True)

        # ⚠️ Low confidence warning
        if confidence < 50:
            st.warning("⚠️ Low confidence prediction. Try a clearer image.")

        # 📊 Confidence bar (FIXED)
        st.markdown("### 📊 Model Confidence")
        progress_value = float(confidence) / 100
        progress_value = max(0.0, min(progress_value, 1.0))
        st.progress(progress_value)

    else:
        st.warning("Upload image and click Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- MODEL INFO --------
st.markdown("""
<div class="model-box">
<b>📌 Model Information</b><br>
• Model: Convolutional Neural Network (CNN)<br>
• Accuracy: ~92%<br>
• Dataset: PlantVillage<br>
</div>
""", unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("---")
st.markdown("🚀 Built with TensorFlow & Streamlit | Smart Agriculture 🌱")