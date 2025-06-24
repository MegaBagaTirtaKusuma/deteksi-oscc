# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # Diperlukan untuk memuat model .keras
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os
import requests # Untuk mengunduh file dari URL

# =====================
# 2. KONFIGURASI MODEL
# =====================
# Direktori tempat model akan disimpan secara lokal
MODEL_DIR = "model"
# Nama file model setelah diunduh (harus .keras sesuai format modelmu)
MODEL_FILE = "model_resnet152.keras"
# Path lengkap ke file model di sistem lokal
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# --- KONFIGURASI GOOGLE DRIVE ---
# GANTI DENGAN ID FILE GOOGLE DRIVE MODEL .KERAS MILIKMU!
# Cara mendapatkan ID: Dari link Google Drive "https://drive.google.com/file/d/INI_ID_FILE_KAMU/view",
# ID-nya adalah "INI_ID_FILE_KAMU".
# Berdasarkan link yang kamu berikan sebelumnya, ID-mu adalah: 1zbtyAu-rV5qkxc362kCt7fdRxKIcuAoS
GOOGLE_DRIVE_FILE_ID = "1zbtyAu-rV5qkxc362kCt7fdRxKIcuAoS"

# URL untuk direct download dari Google Drive.
# '&confirm=t' sangat penting untuk melewati halaman konfirmasi unduhan file besar.
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}&confirm=t"

# =====================
# 3. FUNGSI UNDUH MODEL
# =====================
def download_model_from_drive():
    """
    Mengunduh file model .keras dari Google Drive jika belum ada di lokal.
    Menampilkan progress bar saat mengunduh.
    """
    if not os.path.exists(MODEL_PATH):
        st.warning("🔁 Mengunduh model dari Google Drive. Ini mungkin memerlukan waktu sedikit...")
        os.makedirs(MODEL_DIR, exist_ok=True) # Pastikan folder 'model' ada

        try:
            response = requests.get(MODEL_URL, stream=True)
            # Akan memicu HTTPError jika status kode respons tidak 200 (misal 404 Not Found)
            response.raise_for_status() 

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 8192 # Ukuran chunk untuk unduhan (8 KB)

            # Inisialisasi progress bar di Streamlit
            progress_bar_text = st.empty()
            progress_bar = st.progress(0)
            bytes_downloaded = 0

            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk: # Pastikan chunk data tidak kosong
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size_in_bytes > 0:
                            percent_done = min(int((bytes_downloaded / total_size_in_bytes) * 100), 100)
                            progress_bar.progress(percent_done)
                            progress_bar_text.text(
                                f"Mengunduh: {bytes_downloaded / (1024*1024):.2f} MB / "
                                f"{total_size_in_bytes / (1024*1024):.2f} MB"
                            )
            
            # Bersihkan progress bar setelah selesai
            progress_bar.empty()
            progress_bar_text.empty()
            st.success("✅ Model berhasil diunduh!")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal mengunduh model: {e}")
            st.info("Pastikan ID Google Drive benar, akses file diatur ke 'Anyone with the link', dan file model tidak dihapus.")
            # Hentikan eksekusi Streamlit jika unduhan gagal agar tidak memuat model yang tidak ada/korup
            st.stop() 
    
    return MODEL_PATH

# --- Panggil fungsi unduh model di awal, sebelum mencoba memuatnya ---
downloaded_model_path = download_model_from_drive()

# =====================
# 4. FUNGSI MUAT MODEL
# =====================
@st.cache_resource # Dekorator ini akan membuat model hanya dimuat sekali saja
def load_oscc_model(model_path):
    """
    Memuat model Keras dari file .keras yang sudah diunduh.
    """
    st.info("⏳ Memuat model, mohon tunggu...")
    try:
        # load_model dari TensorFlow/Keras secara otomatis menangani format .keras
        model = load_model(model_path)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Error memuat model: {e}")
        st.info("Pastikan file model .keras tidak korup dan versi TensorFlow/Keras yang digunakan kompatibel. Coba hapus file model di folder 'model' dan jalankan ulang.")
        st.stop() # Hentikan eksekusi jika model gagal dimuat

# --- Panggil fungsi muat model setelah model dipastikan sudah diunduh ---
model = load_oscc_model(downloaded_model_path)

# =====================
# 5. FUNGSI PREDIKSI
# =====================
def predict_oscc(image_file):
    """
    Melakukan prediksi apakah gambar mukosa oral mengandung KANKER (OSCC) atau NORMAL.
    """
    # Buka gambar dan konversi ke RGB (ResNet biasanya expects 3 channel)
    img = Image.open(image_file).convert('RGB')
    # Resize gambar ke ukuran input yang diharapkan model (misal: 224x224 untuk ResNet)
    img = img.resize((224, 224))
    # Konversi gambar ke array NumPy dan normalisasi piksel ke rentang 0-1
    img_array = img_to_array(img) / 255.0
    # Tambahkan dimensi batch. Model berharap input berbentuk (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Lakukan prediksi
    prediction = model.predict(img_array)
    # Ambil probabilitas untuk kelas positif (kanker), asumsi model output 1 nilai
    probability = prediction[0][0] 
    
    # Tentukan label dan keyakinan berdasarkan threshold 0.5
    if probability > 0.5:
        return "KANKER (OSCC)", float(probability)
    else:
        return "NORMAL", float(1 - probability) # Jika tidak kanker, probabilitasnya adalah 1 - probabilitas kanker

# =====================
# 6. UI UTAMA APLIKASI
# =====================
st.markdown("<h1 style='text-align: center;'>Deteksi Oral Squamous Cell Carcinoma (OSCC)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mukosa oral untuk memeriksa apakah terdapat kanker atau normal.</p>", unsafe_allow_html=True)

# Widget untuk mengunggah file gambar
uploaded_file = st.file_uploader("Pilih gambar mukosa oral...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar yang diunggah di tengah kolom
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(uploaded_file, caption='Gambar yang Diunggah', use_column_width=True)
    st.write("") # Memberi spasi kosong untuk tampilan yang lebih rapi

    # Tombol untuk memulai proses deteksi
    if st.button("Lakukan Deteksi"):
        with st.spinner('Menganalisis gambar...'):
            # Panggil fungsi prediksi
            label, confidence = predict_oscc(uploaded_file)
            time.sleep(1) # Memberi sedikit jeda agar animasi spinner terlihat

        st.subheader("Hasil Deteksi:")
        if label == "KANKER (OSCC)":
            st.error(f"⚠️ **TERDETEKSI: {label}**")
            st.write(f"Tingkat Keyakinan: **{confidence*100:.2f}%**")
            st.info("⚠️ Penting: Hasil ini adalah perkiraan. Selalu konsultasikan dengan profesional medis untuk diagnosa dan penanganan yang akurat.")
        else:
            st.success(f"✅ **TERDETEKSI: {label}**")
            st.write(f"Tingkat Keyakinan: **{confidence*100:.2f}%**")
            st.info("Penting: Terus lakukan pemeriksaan rutin dan jaga kesehatan mulut.")

# Informasi disclaimer di bagian bawah
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Aplikasi ini hanya untuk tujuan demonstrasi dan tidak menggantikan diagnosa medis profesional.</p>", unsafe_allow_html=True)
