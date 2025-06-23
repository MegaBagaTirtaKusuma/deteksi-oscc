# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os
import gdown
import base64
from io import BytesIO

# =====================
# 2. KONFIGURASI HALAMAN
# =====================
st.set_page_config(
    page_title="Deteksi Katarak",
    page_icon="👁️",
    layout="centered"
)

# =====================
# 3. FUNGSI UTAMA
# =====================
# Fungsi untuk mengunduh model
def download_model():
    model_path = 'model/final_model_100.h5'
    if not os.path.exists(model_path):
        os.makedirs('model', exist_ok=True)
        url = 'https://drive.google.com/uc?id=1dAQvcVqh-THln51V_lamzlTBrFHsxmwe'
        gdown.download(url, model_path, quiet=False)
    return model_path

# Fungsi untuk memuat model
@st.cache_resource
def load_eye_model():
    try:
        model_path = download_model()
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# Fungsi untuk prediksi
def predict_eye(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    return ("NORMAL", float(probability)) if probability > 0.5 else ("KATARAK", float(1 - probability))

# =====================
# 4. LOAD MODEL
# =====================
model = load_eye_model()
if model is None:
    st.stop()

# =====================
# 5. UI UTAMA
# =====================
# Judul Aplikasi
st.markdown("<h1 style='text-align: center;'>Deteksi Katarak</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mata untuk memeriksa apakah terdapat katarak</p>", unsafe_allow_html=True)

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar mata...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(
            f"""
            <div style='display: flex; justify-content: center; width: 100%;'>
                <img src='data:image/png;base64,{img_str}' style='max-width:100%; height:auto; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.1);' alt='Gambar yang Diunggah'/>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Lakukan prediksi
    with st.spinner('Menganalisis...'):
        label, confidence = predict_eye(uploaded_file)
        time.sleep(1)

    st.success('Analisis selesai!')

    # Tampilkan hasil
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hasil", label)
    with col2:
        st.metric("Tingkat Kepercayaan", f"{confidence*100:.2f}%")

    # Saran berdasarkan hasil
    if label == "KATARAK":
        st.warning("Terdeteksi kemungkinan katarak. Silakan konsultasi dengan dokter spesialis mata.")
    else:
        st.info("Tidak terdeteksi katarak. Tetap jaga kesehatan mata dengan pemeriksaan rutin.")

# =====================
# 6. CSS GLOBAL
# =====================
st.markdown(
    """
    <style>
    @media (max-width: 600px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-size: 1.2em !important;
        }
        .stMetric {
            font-size: 1em !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
) 