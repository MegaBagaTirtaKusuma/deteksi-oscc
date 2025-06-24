# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
import tensorflow as tf
import os
import requests

# =====================
# 2. KONFIGURASI MODEL UNTUK H5
# =====================
MODEL_DIR = "model_h5" # Direktori terpisah untuk contoh ini
MODEL_FILE = "model_resnet152.h5" # Ekstensi .h5 untuk format HDF5
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
# Menggunakan URL model yang sama, tetapi akan disimpan sebagai .h5
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/resolve/main/model_resnet152_bs8.keras"


# =====================
# 3. FUNGSI UNDUH MODEL
# =====================
@st.cache_resource
def download_model_h5():
    """
    Downloads the model file from the specified URL and saves it as .h5.
    Caches the download result to prevent re-downloading.
    """
    if not os.path.exists(MODEL_PATH):
        st.warning(f"🔁 Mengunduh model HDF5 dari Hugging Face ke {MODEL_PATH}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            with open(MODEL_PATH, 'wb') as f: # Save with .h5 extension
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("✅ Model HDF5 berhasil diunduh!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal mengunduh model HDF5: {e}. Pastikan URL benar dan ada koneksi internet.")
            return None
    else:
        st.info("Model HDF5 sudah tersedia secara lokal.")
    return MODEL_PATH

# =====================
# 4. FUNGSI MUAT MODEL H5
# =====================
@st.cache_resource
def load_and_cache_model_h5(model_path):
    """
    Loads and caches the HDF5 model using tf.keras.models.load_model.
    """
    if model_path is None:
        st.error("Tidak dapat memuat model karena file model tidak ditemukan atau unduhan gagal.")
        return None

    st.info("🧠 Memuat model HDF5... ini mungkin memerlukan waktu beberapa detik.")
    try:
        # tf.keras.models.load_model can load .h5 (HDF5) format
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model HDF5 berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model HDF5: {e}")
        st.error("Pastikan file model tidak valid atau rusak, atau versi TensorFlow/Keras Anda tidak kompatibel.")
        st.error("Rekomendasi: Coba hapus folder 'model_h5' dan jalankan ulang.")
        return None

# --- GLOBAL MODEL LOADING (H5) ---
downloaded_model_path_h5 = download_model_h5()
model_h5 = load_and_cache_model_h5(downloaded_model_path_h5)


# =====================
# 5. UI UTAMA
# =====================
st.markdown("<h2 style='text-align: center;'>Contoh Pemuatan Model Keras (.h5)</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ini menunjukkan cara memuat model yang disimpan dalam format HDF5 (.h5).</p>", unsafe_allow_html=True)

if model_h5:
    st.success("Model .h5 siap digunakan!")
    st.write("Anda sekarang dapat melanjutkan dengan prediksi atau operasi lain menggunakan `model_h5`.")
else:
    st.error("Model .h5 gagal dimuat. Harap periksa pesan kesalahan di atas.")

# Contoh sederhana penggunaan (hanya untuk menunjukkan model dimuat)
if st.button("Tampilkan Ringkasan Model H5"):
    if model_h5:
        st.text("Ringkasan Model H5:")
        model_h5.summary(print_fn=lambda x: st.text(x))
    else:
        st.warning("Model H5 belum dimuat.")

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
