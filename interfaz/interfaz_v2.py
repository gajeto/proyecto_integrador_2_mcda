import streamlit as st
import pandas as pd
import numpy as np
import json
from pycaret.classification import load_model, predict_model
import plotly.express as px
from collections import Counter
from fpdf import FPDF

# --- Configuración general de la app ---
st.set_page_config(page_title="Predicción de Diagnóstico Médico", page_icon="🩺", layout="centered")
st.title("🩺 Evaluación de Síntomas y Predicción de Diagnóstico")
st.markdown("Completa la información y selecciona los síntomas relacionados para obtener un diagnóstico probable.")

# --- Carga de síntomas ---
@st.cache_resource
def load_symptoms():
    with open('lista_sintomas.txt', 'r') as f:
        symptoms = eval(f.read())
    return symptoms

symptoms_list = load_symptoms()

# --- Carga de modelos ---
@st.cache_resource
def load_all_models():
    model_paths = [
        'MODELOS/modelo_lr',
        'MODELOS/modelo_gauss',
        'MODELOS/modelo_discr',
        'MODELOS/modelo_knn',
        'MODELOS/modelo_tree'
    ]
    models = [load_model(path) for path in model_paths]
    return models

models = load_all_models()
models_names = ['Regresión', 'Gaussiano', 'Discriminante', 'KNN', 'Arból']

# --- Carga de descripciones ---
@st.cache_resource
def load_descriptions():
    with open('diagnosis_descriptions.json', 'r') as f:
        descriptions = json.load(f)
    return descriptions

diagnosis_descriptions = load_descriptions()

# --- Formulario de ingreso del paciente ---
with st.form("patient_info"):
    st.header("📝 Datos del Paciente")
    patient_name = st.text_input("Nombre Completo")
    patient_id = st.text_input("Identificación (opcional)")
    patient_age = st.number_input("Edad", min_value=0, max_value=120, value=30)
    patient_sex = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro"])
    st.header("💊 Selección de Síntomas")
    selected_symptoms = st.multiselect("Selecciona los síntomas:", options=symptoms_list)
    submitted = st.form_submit_button("🔎 Realizar Predicción")

# --- Lógica de predicción ---
if submitted:
    if not selected_symptoms or not patient_name or not patient_age:
        st.warning("⚠️ Por favor completa todos los campos obligatorios.")
    else:
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
        input_df = pd.DataFrame([input_vector], columns=symptoms_list)

        predictions = []
        confidences = []

        for i, model in enumerate(models):
            try:
                prediction_result = predict_model(model, data=input_df)
                label_column = 'Label' if 'Label' in prediction_result.columns else 'prediction_label'
                score_column = 'Score' if 'Score' in prediction_result.columns else 'prediction_score'

                predicted_label = prediction_result.loc[0, label_column]
                predicted_score = prediction_result.loc[0, score_column]

                predictions.append(predicted_label)
                confidences.append(predicted_score)
            except Exception as e:
                st.error(f"Error en el modelo {i+1}: {e}")

        # Mostrar resultados enriquecidos
        st.subheader("📊 Resultados de los modelos:")
        df_results = pd.DataFrame({
            'Modelo': [f'{i}' for i in models_names],
            'Diagnóstico': predictions,
            'Confianza': [f"{c*100:.2f}%" for c in confidences],
            'Confianza_num': confidences
        })
        st.dataframe(df_results[['Modelo', 'Diagnóstico', 'Confianza']])

        fig = px.bar(df_results, x='Modelo', y='Confianza_num', color='Diagnóstico',
                     text='Confianza', labels={'Confianza_num':'Confianza'}, height=400)
        st.plotly_chart(fig)

        counter = Counter(predictions)
        most_common = counter.most_common(1)[0]

        if most_common[1] >= 3:
            final_diagnosis = most_common[0]
            st.success(f"✅ Diagnóstico final por votación: **{final_diagnosis}** (consenso de {most_common[1]} modelos)")
        else:
            idx_max_conf = np.argmax(confidences)
            final_diagnosis = predictions[idx_max_conf]
            st.success(f"✅ Diagnóstico final por mayor confianza: **{final_diagnosis}**")

        description = diagnosis_descriptions.get(final_diagnosis, "Descripción no disponible.")
        st.info(f"📝 **Descripción:** {description}")

        st.subheader("📚 Diagnósticos alternativos sugeridos:")
        for diag, count in counter.most_common(3):
            st.write(f"- **{diag}**: {count} modelo(s)")

        if max(confidences) < 0.6:
            st.warning("⚠️ La confianza en el diagnóstico es moderada. Se recomienda consultar un profesional médico.")

        st.subheader("🔍 Detalle por modelo:")
        for i, (pred, conf) in enumerate(zip(predictions, confidences), 1):
            st.write(f"Modelo {i}: **{pred}** ({conf*100:.2f}%)")

        # --- Generar PDF ---
        st.subheader("📄 Generar Reporte en PDF")

        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", 'B', 16)
                self.cell(0, 10, "Reporte de Diagnóstico", ln=True, align="C")
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", 'I', 8)
                self.cell(0, 10, f"Página {self.page_no()}", 0, 0, 'C')

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, f"Nombre: {patient_name}", ln=True)
        pdf.cell(0, 10, f"ID: {patient_id}", ln=True)
        pdf.cell(0, 10, f"Edad: {patient_age}   Sexo: {patient_sex}", ln=True)
        pdf.ln(5)

        pdf.cell(0, 10, f"Diagnóstico Final: {final_diagnosis}", ln=True)
        pdf.multi_cell(0, 10, f"Descripción: {description}")
        pdf.ln(5)

        pdf.cell(0, 10, "Síntomas asociados:", ln=True)
        for symptom in selected_symptoms:
            pdf.cell(0, 8, f"- {symptom}", ln=True)
        pdf.ln(5)

        pdf.cell(0, 10, "Confianza de los especialistas:", ln=True)
        for i, (pred, conf) in enumerate(zip(predictions, confidences), 1):
            pdf.cell(0, 8, f"Especialista {i}: {pred} ({conf*100:.2f}%)", ln=True)

        pdf_bytes = pdf.output(dest='S').encode('latin1')

        st.download_button(
            label="📥 Descargar PDF",
            data=pdf_bytes,
            file_name="diagnostico_paciente.pdf",
            mime="application/pdf"
        )
