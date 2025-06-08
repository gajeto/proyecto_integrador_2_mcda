import gzip
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    with gzip.open('MODELOS/modelo_gauss_gz.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el modelo al iniciar la app
modelo = load_model()
print('modelo descimori', modelo)