return model


# =====================
# LOAD MODEL SEKALI (CACHE)
# =====================
@st.cache_resource
def get_model():
    return load_custom_model(download_model())

model = get_model()



# =====================
