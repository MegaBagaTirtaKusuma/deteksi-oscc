# ==============================================================================
# Deteksi Oral Squamous Cell Carcinoma (OSCC)
# Aplikasi Streamlit dengan model Deep Learning dari Hugging Face
# ==============================================================================

# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
from PIL import Image # Untuk manipulasi gambar
import numpy as np # Untuk operasi numerik
import tensorflow as tf # Framework Deep Learning
from tensorflow.keras.models import load_model # Khusus untuk memuat model .keras
from tensorflow.keras.preprocessing.image import img_to_array # Untuk konversi gambar ke array
import time # Untuk memberikan jeda waktu (misal: di spinner)
import os # Untuk operasi sistem file (membuat folder, mengecek path)
import requests # Untuk mengunduh model dari URL

# =====================
# 2. KONFIGURASI MODEL
# =====================
# Direktori lokal tempat model akan disimpan setelah diunduh
MODEL_DIR = "model"
# Nama file model yang akan disimpan secara lokal (sesuai dengan di Hugging Face)
MODEL_FILE = "model_resnet152_bs32.keras"
# Path lengkap ke file model di sistem lokal
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# URL unduhan langsung model dari Hugging Face Hub
# Ini adalah URL yang kamu berikan: bagastk/deteksi-oscc/resolve/main/model_resnet152_bs32.keras
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/resolve/main/model_resnet152_bs32.keras"

# =====================
# 3. FUNGSI UNDUH MODEL
# =====================
def download_model_from_huggingface():
    """
    Mengunduh file model .keras dari Hugging Face Hub jika belum ada di lokal.
    Menampilkan progress bar saat mengunduh.
    """
    if not os.path.exists(MODEL_PATH):
        st.warning("🔁 Mengunduh model dari Hugging Face. Ini mungkin memerlukan waktu sedikit...")
        os.makedirs(MODEL_DIR, exist_ok=True) # Pastikan folder 'model' ada

        try:
            response = requests.get(MODEL_URL, stream=True)
            # Memastikan respons HTTP sukses (status kode 200)
            response.raise_for_status()

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 8192 # Ukuran chunk untuk unduhan (8 KB)

            # Inisialisasi progress bar dan teks di Streamlit
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
            st.info("Pastikan URL model benar dan dapat diakses. Periksa koneksi internet Anda.")
            # Hentikan eksekusi Streamlit jika unduhan gagal
            st.stop() 
    
    return MODEL_PATH

# --- Panggil fungsi unduh model di awal eksekusi aplikasi ---
downloaded_model_path = download_model_from_huggingface()

# =====================
# 4. FUNGSI MUAT MODEL
# =====================
@st.cache_resource # Dekorator ini memastikan model hanya dimuat sekali saja
def load_oscc_model(model_path):
    """
    Memuat model Keras dari file .keras yang sudah diunduh.
    """
    st.info("⏳ Memuat model, mohon tunggu sebentar...")
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
    # Buka gambar menggunakan PIL dan konversi ke RGB (ResNet biasanya expects 3 channel)
    img = Image.open(image_file).convert('RGB')
    # Resize gambar ke ukuran input yang diharapkan model (224x224 untuk ResNet)
    img = img.resize((224, 224))
    # Konversi gambar ke array NumPy dan normalisasi nilai piksel ke rentang 0-1
    img_array = img_to_array(img) / 255.0
    # Tambahkan dimensi batch. Model berharap input berbentuk (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Lakukan prediksi menggunakan model
    prediction = model.predict(img_array)
    # Ambil probabilitas untuk kelas positif (kanker), asumsi model output 1 nilai
    probability = prediction[0][0] 
    
    # Tentukan label dan tingkat keyakinan berdasarkan threshold 0.5
    if probability > 0.5:
        return "KANKER (OSCC)", float(probability)
    else:
        # Jika bukan kanker, probabilitasnya adalah 1 - probabilitas kanker
        return "NORMAL", float(1 - probability)

# =====================
# 6. UI UTAMA APLIKASI
# =====================
st.markdown("<h1 style='text-align: center;'>Deteksi Oral Squamous Cell Carcinoma (OSCC)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mukosa oral untuk memeriksa apakah terdapat kanker atau normal.</p>", unsafe_allow_html=True)

# Widget untuk mengunggah file gambar
uploaded_file = st.file_uploader("Pilih gambar mukosa oral...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar yang diunggah di tengah kolom
    col1_img, col2_img, col3_img = st.columns([1,2,1])
    with col2_img:
        st.image(uploaded_file, caption='Gambar yang Diunggah', use_container_width=True)
    st.write("") # Memberi spasi kosong untuk tampilan yang lebih rapi

    # --- Bagian untuk meratakan tombol "Lakukan Deteksi" ---
    # Menggunakan rasio kolom yang berbeda untuk memaksa tombol terlihat lebih di tengah
    # [spasi_kiri_kosong, kolom_untuk_tombol, spasi_kanan_kosong]
    # Rasio [3, 2, 3] akan memberikan ruang kosong yang lebih besar di samping,
    # membuat kolom tengah relatif lebih sempit dan tombol terlihat di tengah.
    col_left_btn, col_center_btn, col_right_btn = st.columns([3, 2, 3])
    with col_center_btn:
        run_detection = st.button("Lakukan Deteksi")

    if run_detection:
        with st.spinner('Menganalisis gambar...'):
            label, confidence = predict_oscc(uploaded_file)
            time.sleep(1)

        st.write("") # Spasi kosong sebelum hasil
        
        # --- Bagian untuk meratakan subheader dan hasil deteksi (sudah pakai markdown untuk center) ---
        st.markdown("<h3 style='text-align: center;'>Hasil Deteksi:</h3>", unsafe_allow_html=True)
        
        if label == "KANKER (OSCC)":
            st.markdown(f"<div style='text-align: center; color: red; font-weight: bold;'>⚠️ TERDETEKSI: {label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center;'>Tingkat Keyakinan: <b>{confidence*100:.2f}%</b></div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: orange; font-size: small;'>⚠️ Penting: Hasil ini adalah perkiraan. Selalu konsultasikan dengan profesional medis untuk diagnosa dan penanganan yang akurat.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: center; color: green; font-weight: bold;'>✅ TERDETEKSI: {label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center;'>Tingkat Keyakinan: <b>{confidence*100:.2f}%</b></div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: gray; font-size: small;'>Penting: Terus lakukan pemeriksaan rutin dan jaga kesehatan mulut.<div/>", unsafe_allow_html=True) # Tambah penutup div

# Informasi disclaimer di bagian bawah aplikasi
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Aplikasi ini hanya untuk tujuan demonstrasi dan tidak menggantikan diagnosa medis profesional.</p>", unsafe_allow_html=True)
