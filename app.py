# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # Hanya perlu load_model untuk .keras
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os
import requests # Untuk mengunduh dari URL

# =====================
# 2. KONFIGURASI MODEL
# =====================
MODEL_DIR = "model" # Folder untuk menyimpan model yang diunduh
MODEL_FILE = "model_resnet152_bs32.keras" # Nama file model setelah diunduh (pastikan ekstensi .keras)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# --- KONFIGURASI GOOGLE DRIVE UNTUK FILE .KERAS ---
# GANTI DENGAN ID FILE GOOGLE DRIVE MODEL .KERAS MILIKMU!
# Cara mendapatkan ID: Dari link Google Drive "https://drive.google.com/file/d/INI_ID_FILE_KAMU/view",
# ID-nya adalah "INI_ID_FILE_KAMU".
GOOGLE_DRIVE_FILE_ID = "1zbtyAu-rV5qkxc362kCt7fdRxKIcuAoS" # <--- INI ID YANG HARUS KAMU GUNAKAN

# URL untuk direct download dari Google Drive, termasuk bypassing konfirmasi file besar
# Parameter '&confirm=t' sangat penting untuk unduhan langsung tanpa interupsi.
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}&confirm=t"

# Sisa kode lainnya tetap sama seperti yang sudah saya berikan sebelumnya.
# Pastikan tidak ada kode yang menangani file .h5 atau JSON di fungsi load_model.
# =====================
# 3. UNDUH MODEL
# =====================
def download_model():
    """Mengunduh model dari URL (Google Drive) jika belum ada secara lokal."""
    if not os.path.exists(MODEL_PATH):
        st.warning("🔁 Mengunduh model dari Google Drive. Ini mungkin memerlukan waktu sedikit...")
        os.makedirs(MODEL_DIR, exist_ok=True) # Buat folder 'model' jika belum ada
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Akan memicu HTTPError untuk respons status kode yang buruk (misal 404, 500)

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 8192 # Ukuran chunk untuk unduhan

            # Tampilkan progress bar
            progress_bar_text = st.empty()
            progress_bar = st.progress(0)
            bytes_downloaded = 0

            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk: # Pastikan chunk tidak kosong
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size_in_bytes > 0:
                            percent_done = min(int((bytes_downloaded / total_size_in_bytes) * 100), 100)
                            progress_bar.progress(percent_done)
                            progress_bar_text.text(f"Mengunduh: {bytes_downloaded / (1024*1024):.2f} MB / {total_size_in_bytes / (1024*1024):.2f} MB")
            progress_bar.empty() # Hilangkan progress bar dan teks setelah selesai
            progress_bar_text.empty()
            st.success("✅ Model berhasil diunduh!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal mengunduh model: {e}")
            st.info("Pastikan ID Google Drive benar dan akses file diatur ke 'Anyone with the link'.")
            st.stop() # Hentikan eksekusi Streamlit jika unduhan gagal
    return MODEL_PATH

# --- Panggil fungsi download_model() terlebih dahulu untuk memastikan model tersedia ---
model_path_downloaded = download_model()

# =====================
# 4. LOAD MODEL (Untuk format .keras)
# =====================
# Menggunakan st.cache_resource agar model hanya dimuat sekali saat aplikasi berjalan
@st.cache_resource
def load_keras_model_cached(keras_path):
    """Memuat model dari format .keras."""
    st.info("⏳ Memuat model, mohon tunggu...")
    try:
        # load_model() dari tensorflow.keras.models secara otomatis menangani format .keras
        model = load_model(keras_path)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e: # Tangkap semua jenis error yang mungkin terjadi saat loading
        st.error(f"❌ Error memuat model: {e}")
        st.info("Pastikan file model .keras tidak korup dan versi TensorFlow/Keras yang digunakan kompatibel.")
        st.stop() # Hentikan eksekusi jika model tidak bisa dimuat

# --- Panggil fungsi load_keras_model_cached setelah model dipastikan ada ---
model = load_keras_model_cached(model_path_downloaded)


# =====================
# 5. FUNGSI PREDIKSI
# =====================
def predict_oscc(image):
    """Melakukan prediksi pada gambar yang diberikan."""
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224)) # Ukuran input yang diharapkan oleh ResNet152
    img_array = img_to_array(img) / 255.0 # Normalisasi piksel ke rentang 0-1
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch (1, 224, 224, 3)
    
    prediction = model.predict(img_array)
    probability = prediction[0][0] # Ambil probabilitas untuk kelas positif (kanker)
    
    # Menentukan hasil berdasarkan threshold 0.5
    if probability > 0.5:
        return "KANKER (OSCC)", float(probability)
    else:
        return "NORMAL", float(1 - probability) # Probabilitas untuk kelas NORMAL

# =====================
# 6. UI UTAMA APLIKASI
# =====================
st.markdown("<h1 style='text-align: center;'>Deteksi Oral Squamous Cell Carcinoma (OSCC)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mukosa oral untuk memeriksa apakah terdapat kanker atau normal.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih gambar mukosa oral...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar yang diunggah
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(uploaded_file, caption='Gambar yang Diunggah', use_column_width=True)
    st.write("") # Memberi spasi kosong

    # Tombol untuk prediksi
    if st.button("Lakukan Deteksi"):
        with st.spinner('Menganalisis gambar...'):
            label, confidence = predict_oscc(uploaded_file)
            time.sleep(1) # Memberi sedikit jeda agar spinner terlihat lebih baik

        st.subheader("Hasil Deteksi:")
        if label == "KANKER (OSCC)":
            st.error(f"⚠️ **TERDETEKSI: {label}**")
            st.write(f"Tingkat Keyakinan: **{confidence*100:.2f}%**")
            st.info("⚠️ Disarankan untuk berkonsultasi dengan profesional medis untuk diagnosa lebih lanjut.")
        else:
            st.success(f"✅ **TERDETEKSI: {label}**")
            st.write(f"Tingkat Keyakinan: **{confidence*100:.2f}%**")
            st.info("Penting untuk tetap melakukan pemeriksaan rutin.")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Aplikasi ini hanya untuk tujuan demonstrasi dan tidak menggantikan diagnosa medis profesional.</p>", unsafe_allow_html=True)
