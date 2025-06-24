# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
# Hapus import model_from_json, img_to_array jika tf.keras sudah mencakupnya secara internal
# Kita tetap butuh img_to_array, tapi load_model dari tf.keras saja sudah cukup
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os
import requests
import base64
from io import BytesIO
# import h5py # TIDAK PERLU lagi karena kita akan pakai tf.keras.models.load_model untuk .keras
# import json # TIDAK PERLU lagi karena kita akan pakai tf.keras.models.load_model untuk .keras

# =====================
# 2. KONFIGURASI MODEL
# =====================
MODEL_DIR = "model"
# Ubah ekstensi file lokal agar sesuai dengan format .keras
MODEL_FILE = "model_resnet152_bs8.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
# Ini adalah URL yang benar dari Hugging Face untuk file LFS
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/resolve/main/model_resnet152_bs8.keras"


# =====================
# 3. UNDUH MODEL
# =====================
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"🔁 Mengunduh model dari Hugging Face ke {MODEL_PATH}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Cek jika ada error HTTP (misal: 404)
            
            # Mendapatkan ukuran total dari header Content-Length
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            if total_size_in_bytes == 0:
                st.warning("⚠️ Tidak dapat menentukan ukuran file dari header Content-Length. Unduhan mungkin tidak akurat.")
                # Lanjutkan unduhan tanpa progress bar yang akurat

            block_size = 8192 # bytes
            progress_bar = st.progress(0)
            downloaded_size = 0

            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size_in_bytes > 0: # Hanya update progress jika total_size diketahui
                            progress = min(int((downloaded_size / total_size_in_bytes) * 100), 100)
                            progress_bar.progress(progress)
            
            # Validasi ukuran file setelah unduh
            actual_file_size = os.path.getsize(MODEL_PATH)
            if total_size_in_bytes > 0 and actual_file_size != total_size_in_bytes:
                st.error(f"❌ Ukuran file yang diunduh tidak cocok! Diharapkan: {total_size_in_bytes} bytes, Aktual: {actual_file_size} bytes. File mungkin rusak.")
                os.remove(MODEL_PATH) # Hapus file yang rusak agar diunduh ulang
                st.stop()
            elif actual_file_size == 0:
                st.error(f"❌ File yang diunduh kosong ({actual_file_size} bytes). File mungkin rusak.")
                os.remove(MODEL_PATH)
                st.stop()
            else:
                st.success("✅ Model berhasil diunduh!")
                st.info(f"Ukuran file yang diunduh: {actual_file_size / (1024 * 1024):.2f} MB.") # Tampilkan dalam MB

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal mengunduh model. Pastikan URL benar dan ada koneksi internet. Detail: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH) # Hapus file yang tidak lengkap/rusak
            st.stop() # Hentikan aplikasi jika unduh gagal
    else:
        actual_file_size = os.path.getsize(MODEL_PATH)
        st.info(f"💡 Model sudah ada secara lokal di {MODEL_PATH}.")
        st.info(f"Ukuran file lokal: {actual_file_size / (1024 * 1024):.2f} MB.")
    return MODEL_PATH

# =====================
# 4. LOAD MODEL MENGGUNAKAN TF.KERAS.MODELS.LOAD_MODEL
# =====================
# Fungsi ini sekarang jauh lebih sederhana dan lebih robust untuk format .keras
def load_custom_model(model_path):
    st.info(f"⏳ Memuat model dari: {model_path}")
    try:
        # Gunakan tf.keras.models.load_model langsung, ini lebih tepat untuk format .keras
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model. Pastikan file '{model_path}' adalah model Keras (.keras) yang valid.")
        st.error(f"Detail error: {e}")
        st.warning(f"Coba hapus file '{model_path}' secara manual jika error terus berlanjut dan restart aplikasi.")
        st.stop() # Hentikan aplikasi jika gagal memuat model


# =====================
# 5. FUNGSI PREDIKSI
# =====================
# Penting: Fungsi predict_oscc harus menerima objek model sebagai argumen!
def predict_oscc(image, model_obj): # Ganti 'model' menjadi 'model_obj' untuk menghindari konflik nama
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model_obj.predict(img_array) # Gunakan model_obj yang dilewatkan
    probability = prediction[0][0] 
    
    return ("KANKER (OSCC)", float(probability)) if probability > 0.5 else ("NORMAL", float(1 - probability))

# =====================
# 6. UI UTAMA
# =====================
st.markdown("<h1 style='text-align: center;'>Deteksi Oral Squamous Cell Carcinoma (OSCC)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mukosa oral untuk memeriksa apakah terdapat kanker</p>", unsafe_allow_html=True)

# Unduh dan muat model sekali saja
# Gunakan st.cache_resource untuk menghindari pengunduhan dan pemuatan berulang
@st.cache_resource
def get_model_cached(): # Ubah nama fungsi untuk menghindari kebingungan
    model_path = download_model()
    model = load_custom_model(model_path)
    return model

# Muat model di awal aplikasi
# Ini akan memanggil get_model_cached() pertama kali, dan akan menggunakan cache selanjutnya
model_loaded = get_model_cached() # Simpan model di variabel dengan nama yang jelas

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
        # Lewatkan objek model_loaded ke fungsi predict_oscc
        label, confidence = predict_oscc(uploaded_file, model_loaded)
        time.sleep(1)

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
