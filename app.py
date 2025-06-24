# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
import tensorflow as tf
import os
import requests

# =====================
# 2. KONFIGURASI MODEL UNTUK .KERAS
# =====================
MODEL_DIR = "model_keras" # Direktori terpisah untuk contoh ini
MODEL_FILE = "model_resnet152_bs8.keras" # Ekstensi .keras untuk format native Keras v3
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/resolve/main/model_resnet152_bs8.keras"


# =====================
# 3. FUNGSI UNDUH MODEL
# =====================
@st.cache_resource
def download_model_keras():
    """
    Downloads the model file from the specified URL and saves it as .keras.
    Caches the download result to prevent re-downloading.
    """
    if not os.path.exists(MODEL_PATH):
        st.warning(f"🔁 Mengunduh model Keras native dari Hugging Face ke {MODEL_PATH}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            with open(MODEL_PATH, 'wb') as f: # Save with .keras extension
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("✅ Model Keras native berhasil diunduh!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal mengunduh model Keras native: {e}. Pastikan URL benar dan ada koneksi internet.")
            return None
    else:
        st.info("Model Keras native sudah tersedia secara lokal.")
    return MODEL_PATH

# =====================
# 4. FUNGSI MUAT MODEL .KERAS
# =====================
@st.cache_resource
def load_and_cache_model_keras(model_path):
    """
    Loads and caches the .keras model using tf.keras.models.load_model.
    """
    if model_path is None:
        st.error("Tidak dapat memuat model karena file model tidak ditemukan atau unduhan gagal.")
        return None

    st.info("🧠 Memuat model Keras native... ini mungkin memerlukan waktu beberapa detik.")
    try:
        # tf.keras.models.load_model can load .keras (zip archive) format
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model Keras native berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model Keras native: {e}")
        st.error("Pastikan file model tidak valid atau rusak, atau versi TensorFlow/Keras Anda tidak kompatibel.")
        st.error("Rekomendasi: Pastikan Anda menggunakan TensorFlow 2.10 atau yang lebih baru. Coba hapus folder 'model_keras' dan jalankan ulang.")
        return None

# --- GLOBAL MODEL LOADING (.KERAS) ---
downloaded_model_path_keras = download_model_keras()
model_keras = load_and_cache_model_keras(downloaded_model_path_keras)


# =====================
# 5. UI UTAMA
# =====================
st.markdown("<h2 style='text-align: center;'>Contoh Pemuatan Model Keras (.keras)</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ini menunjukkan cara memuat model yang disimpan dalam format native Keras (.keras).</p>", unsafe_allow_html=True)

if model_keras:
    st.success("Model .keras siap digunakan!")
    st.write("Anda sekarang dapat melanjutkan dengan prediksi atau operasi lain menggunakan `model_keras`.")
else:
    st.error("Model .keras gagal dimuat. Harap periksa pesan kesalahan di atas.")

# Contoh sederhana penggunaan (hanya untuk menunjukkan model dimuat)
if st.button("Tampilkan Ringkasan Model .keras"):
    if model_keras:
        st.text("Ringkasan Model .keras:")
        model_keras.summary(print_fn=lambda x: st.text(x))
    else:
        st.warning("Model .keras belum dimuat.")

# CSS responsif (sama seperti sebelumnya)
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
