# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
# Hapus import model_from_json, img_to_array jika tf.keras sudah mencakupnya secara internal
from tensorflow.keras.preprocessing.image import img_to_array # Ini masih perlu
import time
import os
import requests
import base64
from io import BytesIO
# import h5py # Tidak perlu lagi jika menggunakan tf.keras.models.load_model langsung
# import json # Tidak perlu lagi jika menggunakan tf.keras.models.load_model langsung

# =====================
# 2. KONFIGURASI MODEL
# =====================
MODEL_DIR = "model"
# Ubah nama file lokal agar sesuai dengan ekstensi .keras jika perlu, atau tetap .h5
# Tapi kontennya akan diperlakukan sebagai format .keras
MODEL_FILE = "model_resnet152_bs8.keras" # Lebih akurat dengan URL sumber
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/raw/main/model_resnet152_bs8.keras"


# =====================
# 3. UNDUH MODEL
# =====================
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"🔁 Mengunduh model dari Hugging Face ke {MODEL_PATH}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Cek jika ada error HTTP
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192 # bytes
            progress_bar = st.progress(0)
            downloaded_size = 0

            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = min(int((downloaded_size / total_size) * 100), 100)
                        progress_bar.progress(progress)
            st.success("✅ Model berhasil diunduh!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal mengunduh model: {e}")
            st.stop() # Hentikan aplikasi jika gagal unduh
    else:
        st.info("💡 Model sudah ada secara lokal. Melewati pengunduhan.")
    return MODEL_PATH

# =====================
# 4. LOAD MODEL MENGGUNAKAN TF.KERAS.MODELS.LOAD_MODEL
# =====================
# Fungsi ini sekarang jauh lebih sederhana dan lebih robust untuk format .keras
def load_custom_model(model_path):
    st.info(f"⏳ Memuat model dari: {model_path}")
    try:
        # Gunakan load_model langsung, ini lebih tepat untuk format .keras
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model. Pastikan file '{model_path}' adalah model Keras yang valid.")
        st.error(f"Detail error: {e}")
        st.stop() # Hentikan aplikasi jika gagal memuat model


# =====================
# 5. FUNGSI PREDIKSI
# =====================
def predict_oscc(image, model):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Lakukan prediksi
    prediction = model.predict(img_array)
    probability = prediction[0][0] # Asumsi ini adalah binary classification (0 atau 1)

    # Menentukan label dan tingkat kepercayaan
    if probability > 0.5:
        return ("KANKER (OSCC)", float(probability))
    else:
        return ("NORMAL", float(1 - probability)) # Jika NORMAL, kepercayaan adalah 1 - probabilitas kanker

# =====================
# 6. UI UTAMA
# =====================
st.markdown("<h1 style='text-align: center;'>Deteksi Oral Squamous Cell Carcinoma (OSCC)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mukosa oral untuk memeriksa apakah terdapat kanker</p>", unsafe_allow_html=True)

# Unduh dan muat model sekali saja menggunakan st.cache_resource
@st.cache_resource
def get_model():
    model_path = download_model()
    model = load_custom_model(model_path)
    return model

# Muat model di awal aplikasi
# Spinner akan muncul secara otomatis karena get_model dipanggil di luar kondisi if uploaded_file
model = get_model()

uploaded_file = st.file_uploader("Pilih gambar OSCC atau Normal...", type=["jpg", "jpeg", "png"])

if uploaded_file:
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

    with st.spinner('🧠 Menganalisis...'):
        label, confidence = predict_oscc(uploaded_file, model)
        time.sleep(1) # Memberi waktu spinner terlihat

    st.success('✅ Analisis selesai!')

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hasil", label)
    with col2:
        st.metric("Tingkat Kepercayaan", f"{confidence*100:.2f}%")

    if label == "KANKER (OSCC)":
        st.warning("⚠️ Terdeteksi kemungkinan OSCC. Disarankan segera konsultasi dengan dokter spesialis.")
    else:
        st.info("✅ Tidak terdeteksi kanker. Tetap periksa secara berkala untuk deteksi dini.")

# =====================
# 7. CSS RESPONSIF
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
