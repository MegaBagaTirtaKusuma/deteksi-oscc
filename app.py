# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # load_model is sufficient for .keras or .h5
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os
import requests
import base64
from io import BytesIO

# =====================
# 2. KONFIGURASI MODEL
# =====================
MODEL_DIR = "model"
MODEL_FILE = "model_resnet152.h5" # Mengubah ekstensi file menjadi .h5
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/resolve/main/model_resnet152_bs8.keras" # URL tetap menggunakan .keras karena itu yang tersedia


# =====================
# 3. UNDUH MODEL
# =====================
# @st.cache_resource will cache the result of this function,
# so the model is downloaded only once.
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("🔁 Mengunduh model dari Hugging Face... Ini mungkin memerlukan waktu beberapa saat.")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Akan menaikkan HTTPError untuk kode status respons yang buruk (4xx atau 5xx)
            with open(MODEL_PATH, 'wb') as f: # Menyimpan sebagai .h5
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("✅ Model berhasil diunduh!") # Tambahkan kembali pesan sukses untuk konfirmasi unduhan
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal mengunduh model: {e}. Pastikan URL benar dan ada koneksi internet.")
            return None # Mengembalikan None jika unduhan gagal
    else:
        st.info("Model sudah tersedia secara lokal.") # Ubah warning ke info
    return MODEL_PATH

# =====================
# 4. LOAD MODEL DENGAN FIX (Menggunakan tf.keras.models.load_model)
# =====================
# @st.cache_resource will cache the model object itself,
# so it's loaded into memory only once.
@st.cache_resource
def load_and_cache_model(model_path):
    if model_path is None: # Jika path model tidak valid (misal, karena unduhan gagal)
        st.error("Tidak dapat memuat model karena file model tidak ditemukan atau unduhan gagal.")
        return None

    st.info("🧠 Memuat model... ini mungkin memerlukan waktu beberapa detik.")
    try:
        # tf.keras.models.load_model dapat memuat format .keras dan .h5
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model berhasil dimuat!") # Tambahkan kembali pesan sukses untuk konfirmasi pemuatan
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.error("Pastikan file model tidak rusak dan kompatibel dengan versi TensorFlow Anda. Coba hapus folder 'model' dan jalankan ulang.")
        return None # Mengembalikan None jika pemuatan gagal

# --- GLOBAL MODEL LOADING ---
# Kode ini akan dieksekusi sekali saat aplikasi dimulai.
# Variabel 'model' akan berisi objek model atau None jika terjadi kegagalan.
downloaded_model_path = download_model()
model = load_and_cache_model(downloaded_model_path)


# =====================
# 5. FUNGSI PREDIKSI
# =====================
def predict_oscc(image):
    # 'model' sudah didefinisikan secara global dan dimuat pada titik ini
    # Tidak perlu memeriksa model di sini, karena sudah diperiksa di bagian UI utama.
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    return ("KANKER (OSCC)", float(probability)) if probability > 0.5 else ("NORMAL", float(1 - probability))

# =====================
# 6. UI UTAMA
# =====================
st.markdown("<h1 style='text-align: center;'>Deteksi Oral Squamous Cell Carcinoma (OSCC)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mukosa oral untuk memeriksa apakah terdapat kanker</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih gambar OSCC atau Normal...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # --- Tambahkan pemeriksaan ini sebelum memanggil predict_oscc ---
    if model is None:
        st.error("Tidak dapat melanjutkan prediksi karena model tidak berhasil dimuat. Silakan coba muat ulang aplikasi atau periksa koneksi internet.")
        st.stop() # Hentikan eksekusi lebih lanjut jika model tidak ada

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
        label, confidence = predict_oscc(uploaded_file)
        time.sleep(1) # Simulasi waktu analisis

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
