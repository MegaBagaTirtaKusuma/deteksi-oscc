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
MODEL_FILE = "model_resnet152.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL = "https://huggingface.co/bagastk/deteksi-oscc/resolve/main/model_resnet152_bs8.keras"


# =====================
# 3. UNDUH MODEL
# =====================
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("🔁 Mengunduh model dari Hugging Face...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return MODEL_PATH

# =====================
# 4. LOAD MODEL DENGAN FIX
# =====================
def load_custom_model(h5_path):
    with h5py.File(h5_path, "r") as f:
        model_config = f.attrs.get("model_config")
        if model_config is None:
            raise ValueError("Model config is missing in HDF5 file.")
        
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        
        model_json = json.loads(model_config)

        # Hapus batch_shape & batch_input_shape
        for layer in model_json["config"]["layers"]:
            layer_config = layer["config"]
            layer_config.pop("batch_input_shape", None)
            layer_config.pop("batch_shape", None)

        # Hapus juga field di model config level atas (opsional tapi aman)
        model_json["config"].pop("batch_input_shape", None)

        cleaned_model_config = json.dumps(model_json)

    try:
        model = model_from_json(cleaned_model_config)
    except ValueError as e:
        st.error("Model error: Kemungkinan format .h5 tidak sepenuhnya kompatibel dengan deserializer JSON.")
        raise e

    model.load_weights(h5_path)
    return model




# =====================
# 5. FUNGSI PREDIKSI
# =====================
def predict_oscc(image):
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
