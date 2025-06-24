# =====================
# 1. IMPORT LIBRARY
# =====================
import streamlit as st
import tensorflow as tf
import os
import requests

# =====================
# Streamlit Page Configuration (PENTING untuk Tampilan)
# =====================
st.set_page_config(
    page_title="Deteksi Oral Squamous Cell Carcinoma (OSCC)",
    page_icon="�",
    layout="wide", # Mengubah layout menjadi lebar untuk tampilan yang lebih luas
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.streamlit.io/docs',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# Aplikasi Deteksi Oral Squamous Cell Carcinoma"
    }
)

# =====================
# 2. KONFIGURASI MODEL UNTUK .KERAS
# =====================
MODEL_DIR = "model_keras" # Direktori untuk model .keras
MODEL_FILE = "model_resnet152_bs8.keras" # Ekstensi .keras untuk format native Keras v3
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/resolve/main/model_resnet152_bs8.keras" # URL model. Pastikan ini adalah model .keras yang valid.

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
st.markdown("<h1 style='text-align: center; color: #2C3E50; margin-bottom: 10px;'>Deteksi Oral Squamous Cell Carcinoma (OSCC)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.1em; margin-bottom: 30px;'>Unggah gambar mukosa oral untuk memeriksa apakah terdapat kanker</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih gambar OSCC atau Normal...", type=["jpg", "jpeg", "png"], accept_multiple_files=False, help="Unggah file gambar dengan format JPG, JPEG, atau PNG.")

# Container untuk menempatkan UI di tengah
if uploaded_file:
    # --- Tambahkan pemeriksaan ini sebelum memanggil predict_oscc ---
    if model_keras is None:
        st.error("Tidak dapat melanjutkan prediksi karena model tidak berhasil dimuat. Silakan coba muat ulang aplikasi atau periksa koneksi internet.")
        st.stop() # Hentikan eksekusi lebih lanjut jika model tidak ada

    # Untuk tampilan gambar di tengah
    image = tf.keras.preprocessing.image.load_img(uploaded_file)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True) # Menggunakan st.image yang lebih Streamlit-native

    st.markdown("<div class='stButton-container' style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    if st.button("Mulai Analisis"):
        with st.spinner('🧠 Menganalisis...'):
            # Dummy prediction for demonstration if model_keras is None
            if model_keras:
                # Assuming your predict_oscc function is in the scope and uses model_keras
                # For this example, we need to adapt the existing predict_oscc
                from PIL import Image # Ensure PIL is imported
                from tensorflow.keras.preprocessing.image import img_to_array
                import numpy as np
                from io import BytesIO

                def predict_oscc_with_keras_model(image_file, loaded_model):
                    img = Image.open(image_file).convert('RGB')
                    img = img.resize((224, 224))
                    img_array = img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    prediction = loaded_model.predict(img_array)
                    probability = prediction[0][0]
                    return ("KANKER (OSCC)", float(probability)) if probability > 0.5 else ("NORMAL", float(1 - probability))

                label, confidence = predict_oscc_with_keras_model(uploaded_file, model_keras)
            else:
                # Fallback for demonstration if model loading truly failed (should be caught by st.error above)
                st.warning("Model belum dimuat. Menggunakan hasil simulasi.")
                # Simple simulation if model is not loaded for some reason
                if "cancer" in uploaded_file.name.lower() or "oscc" in uploaded_file.name.lower():
                    label, confidence = "KANKER (OSCC)", 0.85
                else:
                    label, confidence = "NORMAL", 0.92
            time.sleep(1) # Simulate analysis time

        st.success('✅ Analisis selesai!')

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Hasil", label)
        with col_res2:
            st.metric("Tingkat Kepercayaan", f"{confidence*100:.2f}%")

        if label == "KANKER (OSCC)":
            st.warning("⚠️ Terdeteksi kemungkinan OSCC. Disarankan segera konsultasi dengan dokter spesialis.")
        else:
            st.info("✅ Tidak terdeteksi kanker. Tetap periksa secara berkala untuk deteksi dini.")
    st.markdown("</div>", unsafe_allow_html=True) # Close stButton-container

else:
    st.info("Unggah gambar untuk memulai deteksi.")

# =====================
# 6. CSS Kustom untuk Estetika
# =====================
st.markdown(
    """
    <style>
    /* Import Google Font - Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    body {
        background-color: #f0f2f6; /* Warna latar belakang umum */
    }

    /* Main container styling */
    .stApp {
        background-color: #f0f2f6;
    }

    .css-1d391kg.e1tzin5v1 { /* Selector for main content block */
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #ffffff; /* Latar belakang putih untuk kontainer utama */
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); /* Bayangan lembut */
        margin-bottom: 20px; /* Spasi antar block */
    }

    /* Headings */
    h1 {
        font-weight: 700 !important;
        color: #2C3E50 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }

    h2 {
        font-weight: 600 !important;
        color: #34495E !important;
    }

    p {
        color: #555 !important;
        font-size: 1.05em !important;
        line-height: 1.6;
    }

    /* File uploader button */
    .stFileUploader > div > div > button {
        background-color: #3498DB; /* Warna biru */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1em;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }
    .stFileUploader > div > div > button:hover {
        background-color: #2980B9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stFileUploader label {
        font-weight: 600;
        color: #34495E;
    }

    /* Custom button styling */
    .stButton > button {
        background-color: #2ECC71; /* Warna hijau */
        color: white;
        border-radius: 25px; /* Lebih bulat */
        padding: 12px 30px;
        font-size: 1.1em;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
        border: none;
        cursor: pointer;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #27AE60;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    /* St.image styling */
    .stImage img {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 2px solid #ECF0F1;
        object-fit: contain;
    }
    .stImage > label {
        text-align: center;
        font-style: italic;
        color: #777;
    }


    /* Metrics styling */
    .stMetric {
        background-color: #ECF0F1;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 15px;
    }
    .stMetric > div > div:first-child { /* Label */
        color: #34495E !important;
        font-weight: 600;
        font-size: 1.1em;
    }
    .stMetric > div > div:last-child { /* Value */
        font-weight: 700;
        font-size: 1.8em;
        color: #2C3E50;
    }

    /* Info, Warning, Success boxes */
    .stAlert {
        border-radius: 8px;
        font-size: 1.05em;
        font-weight: 500;
        padding: 15px 20px;
        margin-top: 20px;
    }
    .stAlert.info {
        background-color: #D6ECF7; /* Light blue */
        color: #2196F3; /* Darker blue text */
        border-left: 5px solid #2196F3;
    }
    .stAlert.warning {
        background-color: #FFECB3; /* Light yellow */
        color: #FFA000; /* Darker yellow text */
        border-left: 5px solid #FFA000;
    }
    .stAlert.success {
        background-color: #D4EDDA; /* Light green */
        color: #28A745; /* Darker green text */
        border-left: 5px solid #28A745;
    }

    /* Spinner style */
    .stSpinner > div > div {
        color: #3498DB; /* Blue spinner */
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        h1 {
            font-size: 2em !important;
        }
        h2 {
            font-size: 1.5em !important;
        }
        p {
            font-size: 0.95em !important;
        }
        .stButton > button {
            width: 100%;
            padding: 10px 20px;
            font-size: 1em;
        }
        .stMetric {
            padding: 15px;
        }
        .stMetric > div > div:last-child {
            font-size: 1.5em;
        }
        .stFileUploader > div > div > button {
            width: 100%;
        }
    }

    @media (max-width: 480px) {
        h1 {
            font-size: 1.8em !important;
        }
        p {
            font-size: 0.9em !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)
�
