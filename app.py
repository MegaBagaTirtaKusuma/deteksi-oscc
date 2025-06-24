# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os
import requests
import base64
from io import BytesIO
import h5py
import json

# =====================
# 2. KONFIGURASI MODEL
# =====================
MODEL_DIR = "model"
MODEL_FILE = "model_resnet152_bs32" # Nama file model setelah diunduh
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Ganti dengan ID file Google Drive modelmu.
# Pastikan file di Google Drive sudah diatur "Anyone with the link" (Siapa saja yang memiliki link)
# dan perannya "Viewer" (Pelihat).
# Cara mendapatkan ID: dari link Google Drive "https://drive.google.com/file/d/INI_ID_FILE_KAMU/view",
# ID-nya adalah "INI_ID_FILE_KAMU".
GOOGLE_DRIVE_FILE_ID = "1zbtyAu-rV5qkxc362kCt7fdRxKIcuAoS" # <--- CONTOH ID. GANTI DENGAN ID MILIKMU!

# URL untuk direct download dari Google Drive
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

# =====================
# 3. UNDUH MODEL
# =====================
def download_model():
    """Mengunduh model dari Google Drive jika belum ada."""
    if not os.path.exists(MODEL_PATH):
        st.warning("🔁 Mengunduh model dari Google Drive. Ini mungkin memerlukan waktu...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Akan memicu HTTPError untuk respons status kode yang buruk

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 8192 # 8KB

            progress_bar = st.progress(0)
            bytes_downloaded = 0

            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size_in_bytes > 0:
                            percent_done = min(int((bytes_downloaded / total_size_in_bytes) * 100), 100)
                            progress_bar.progress(percent_done)
            progress_bar.empty() # Menghilangkan progress bar setelah selesai
            st.success("✅ Model berhasil diunduh!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal mengunduh model: {e}")
            st.info("Pastikan ID Google Drive benar dan akses file diatur ke 'Anyone with the link'.")
            st.stop() # Hentikan eksekusi Streamlit jika unduhan gagal
    return MODEL_PATH

# Panggil fungsi download_model() lebih awal untuk memastikan model tersedia
model_path_downloaded = download_model()

# =====================
# 4. LOAD MODEL DENGAN FIX
# =====================
# Hanya load model setelah dipastikan sudah diunduh
@st.cache_resource # Cache model agar tidak load berulang kali
def load_custom_model_cached(h5_path):
    """Memuat model Keras dari file HDF5 dengan penanganan konfigurasi."""
    try:
        with h5py.File(h5_path, "r") as f:
            model_config = f.attrs.get("model_config")
            if model_config is None:
                raise ValueError("Model config is missing in HDF5 file.")
            
            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")
            
            model_json = json.loads(model_config)

            # Hapus batch_shape & batch_input_shape dari konfigurasi layer
            for layer in model_json["config"]["layers"]:
                layer_config = layer["config"]
                layer_config.pop("batch_input_shape", None)
                layer_config.pop("batch_shape", None)

            # Hapus juga field di model config level atas (opsional tapi aman)
            model_json["config"].pop("batch_input_shape", None)

            cleaned_model_config = json.dumps(model_json)

        model = model_from_json(cleaned_model_config)
        model.load_weights(h5_path)
        return model
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as e:
        st.error(f"❌ Error memuat model: {e}")
        st.info("Kemungkinan ada masalah dengan format file model atau korupsi saat pengunduhan.")
        st.stop() # Hentikan eksekusi jika model tidak bisa dimuat

model = load_custom_model_cached(model_path_downloaded)

# =====================
# 5. FUNGSI PREDIKSI
# =====================
def predict_oscc(image):
    """Melakukan prediksi pada gambar yang diberikan."""
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
    prediction = model.predict(img_array)
    probability = prediction[0][0] # Ambil probabilitas untuk kelas positif (kanker)
    
    # Menentukan hasil berdasarkan threshold 0.5
    if probability > 0.5:
        return "KANKER (OSCC)", float(probability)
    else:
        return "NORMAL", float(1 - probability) # Probabilitas untuk kelas NORMAL

# =====================
# 6. UI UTAMA
# =====================
st.markdown("<h1 style='text-align: center;'>Deteksi Oral Squamous Cell Carcinoma (OSCC)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mukosa oral untuk memeriksa apakah terdapat kanker atau normal.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih gambar mukosa oral...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
    st.write("") # Spasi kosong

    # Tombol untuk prediksi
    if st.button("Lakukan Deteksi"):
        with st.spinner('Menganalisis gambar...'):
            label, confidence = predict_oscc(uploaded_file)
            time.sleep(1) # Memberi sedikit jeda agar spinner terlihat

        st.subheader("Hasil Deteksi:")
        if label == "KANKER (OSCC)":
            st.error(f"⚠️ **TERDETEKSI: {label}**")
            st.write(f"Keyakinan: **{confidence*100:.2f}%**")
            st.info("⚠️ Disarankan untuk berkonsultasi dengan profesional medis untuk diagnosa lebih lanjut.")
        else:
            st.success(f"✅ **TERDETEKSI: {label}**")
            st.write(f"Keyakinan: **{confidence*100:.2f}%**")
            st.info("Penting untuk tetap melakukan pemeriksaan rutin.")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Aplikasi ini hanya untuk tujuan demonstrasi dan tidak menggantikan diagnosa medis profesional.</p>", unsafe_allow_html=True)
