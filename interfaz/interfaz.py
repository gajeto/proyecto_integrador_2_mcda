import streamlit as st
import pandas as pd
import numpy as np
import os
from pycaret.classification import load_model, predict_model


# Streamlit layout
st.set_page_config(page_title="Disease Prediction App", page_icon="🩺", layout="centered")

# Load PyCaret pipeline (cached)
@st.cache_resource
def load_pycaret_model():
    model = load_model('modelo_lr')
    return model

# Load symptom list (cached)
@st.cache_resource
def load_symptoms():
    with open('lista_sintomas.txt', 'r') as f:
        content = f.read()
        symptoms = eval(content)
    return symptoms

# Load model and symptoms
model = load_pycaret_model()
symptoms_list = load_symptoms()


st.title("🩺 Modelo predictivo para la evaluación de síntomas generales hospitalarios")
st.markdown("Selecciona los síntomas más relacionados con tu dolencia para darte un probable diagnóstico")

# Sidebar input
with st.sidebar:
    st.header("Síntomas")
    selected_symptoms = st.multiselect(
        "Selecciona sintomas:",
        options=symptoms_list,
        help="Comienza a escribir...s"
    )

    predict_button = st.button("🔎 Predecir diagnóstico")

# Prediction logic
if predict_button:
    if not selected_symptoms:
        st.warning("⚠️ Por favor indica al menos un síntoma")
    else:
        # Build input dataframe for prediction
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
        input_df = pd.DataFrame([input_vector], columns=symptoms_list)
        if 'diseases' in input_df.columns:
            input_df = input_df.drop(columns=['diseases'])


        # Use PyCaret's predict_model
        try:
            prediction_result = predict_model(model, data=input_df)
            label_column = 'Label' if 'Label' in prediction_result.columns else 'prediction_label'
            score_column = 'Score' if 'Score' in prediction_result.columns else 'prediction_score'

            predicted_label = prediction_result.loc[0, label_column]
            predicted_score = prediction_result.loc[0, score_column]

            st.success(f"✅ Posible diagnóstico: **{predicted_label}**")
            st.info(f"Confianza: **{predicted_score*100:.2f}%**")

        except Exception as e:
            st.error(f"❌ Error en la prediccion: {e}")

# Footer
st.markdown("---")
st.caption("Proyecto Integrador 2 - MCDA - EAFIT - 2025")
