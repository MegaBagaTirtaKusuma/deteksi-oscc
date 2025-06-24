# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
import tensorflow as tf
import os
import requests
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import time

# =====================
# 2. KONFIGURASI MODEL UNTUK .KERAS
# =====================
MODEL_DIR = "model_keras"
MODEL_FILE = "model_resnet152_bs8.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/resolve/main/model_resnet152_bs8.keras"

# =====================
# 3. FUNGSI UNDUH MODEL
# =====================
@st.cache_resource
def download_model_keras():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Mengunduh model ke {MODEL_PATH}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model berhasil diunduh!")
        except requests.exceptions.RequestException as e:
            st.error(f"Gagal mengunduh model: {e}")
            return None
    return MODEL_PATH

# =====================
# 4. FUNGSI MUAT MODEL
# =====================
@st.cache_resource
def load_and_cache_model_keras(model_path):
    if model_path is None:
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# =====================
# 5. UI UTAMA
# =====================
st.set_page_config(
    page_title="Deteksi OSCC",
    layout="wide",
)

st.title("Deteksi Oral Squamous Cell Carcinoma (OSCC)")
st.write("Unggah gambar mukosa oral untuk mendeteksi kemungkinan kanker.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

model_path = download_model_keras()
model = load_and_cache_model_keras(model_path)

def predict_oscc(image_file, loaded_model):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = loaded_model.predict(img_array)
    prob = prediction[0][0]
    return ("KANKER (OSCC)", prob) if prob > 0.5 else ("NORMAL", 1 - prob)

if uploaded_file:
    if model is None:
        st.error("Model tidak berhasil dimuat.")
        st.stop()

    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    if st.button("Mulai Analisis"):
        with st.spinner("Menganalisis..."):
            label, confidence = predict_oscc(uploaded_file, model)
            time.sleep(1)
        st.success("Analisis selesai!")
        st.metric("Hasil", label)
        st.metric("Tingkat Kepercayaan", f"{confidence * 100:.2f}%")
        if label == "KANKER (OSCC)":
            st.warning("⚠️ Kemungkinan kanker terdeteksi. Segera konsultasikan dengan dokter.")
        else:
            st.info("✅ Tidak terdeteksi kanker. Tetap lakukan pemeriksaan rutin.")
else:
    st.info("Silakan unggah gambar untuk mulai deteksi.")
