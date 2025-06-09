import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pycaret.classification import load_model, predict_model
import plotly.express as px
from collections import Counter
from fpdf import FPDF

# ================================
# CONFIGURACIÓN DE COLORES
# ================================

# Tema de la interfaz médica
PRIMARY_COLOR = "#2F80ED"  # Azul profesional
SUCCESS_COLOR = "#27AE60"  # Verde confianza
WARNING_COLOR = "#F2994A"  # Naranja de advertencia
ERROR_COLOR = "#EB5757"    # Rojo para errores
BACKGROUND_COLOR = "#F7F9FB"

st.set_page_config(page_title="Predicción Médica", page_icon="🩺", layout="wide")
#st.title("🩺 Sistema de Predicción de Diagnóstico Médico")
st.markdown(
    f"<h1 style='color:{PRIMARY_COLOR};'>🩺 Evaluación de Diagnóstico Médico</h1>", 
    unsafe_allow_html=True
)
st.markdown("Completa los datos y selecciona los síntomas:")


# ================================
# CARGA DE INSUMOS
# ================================

# Obtener el directorio actual del archivo
current_dir = os.path.dirname(os.path.abspath(__file__))

# Subir un nivel al directorio padre
parent_dir = os.path.dirname(current_dir)

# Construir el path absoluto a la imagen en ../insumos/logo_hospital.jpg
logo_path = os.path.join(parent_dir, 'insumos', 'logo_hospital.jpg')

with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(logo_path, width=300)  # replace with your logo path
        st.markdown("<h1 style='text-align: center;'>Predicción Médica</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: gray;'>Proyecto Integrador 2 <br> MCDA <br> EAFIT <br> 2025</h3>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>Especialistas: <br> Juan Pablo Bertel <br> Gustavo Jerez <br> Gustavo Rubio</h1>", unsafe_allow_html=True)


# Expander para info ampliada
with st.expander("ℹ️ Información del modelo de predicción de diagnósticos hospitalarios"):
    st.markdown("En una junta médica, la votación de diagnóstico se realiza de manera colegiada, buscando integrar la experiencia y el criterio de los distintos especialistas presentes. Cada miembro expone su análisis del caso basado en la historia clínica, exámenes complementarios y evidencia científica. Posteriormente, se abre un espacio de discusión donde se contrastan los distintos puntos de vista. Una vez finalizada la deliberación, los participantes emiten su voto, que puede ser abierto o anónimo, dependiendo de los protocolos de la institución. El diagnóstico final se establece por consenso si es posible; en caso de desacuerdo, se adopta la decisión de la mayoría, procurando siempre fundamentar el resultado en criterios clínicos objetivos y en beneficio del paciente.")
    

# Síntomas en inglés (para alimentar el modelo)
@st.cache_resource
def load_symptoms():
    path = os.path.join(parent_dir, 'insumos', 'lista_sintomas.txt')
    with open(path, 'r') as f:
        symptoms = eval(f.read())
    return symptoms

symptoms_list = load_symptoms()

# Traducción de síntomas (EN -> ES)
@st.cache_resource
def load_symptom_translation():
    path = os.path.join(parent_dir, 'insumos', 'sintomas_traducidos.json')
    with open(path, 'r') as f:
        return json.load(f)

symptom_translation = load_symptom_translation()
symptom_translation_rev = {v: k for k, v in symptom_translation.items()}

# Traducción de diagnósticos (EN -> ES)
@st.cache_resource
def load_diagnosis_translation():
    path = os.path.join(parent_dir, 'insumos', 'diagnosticos_traducidos.json')
    with open(path, 'r') as f:
        return json.load(f)

diagnosis_translation = load_diagnosis_translation()

# Descripciones de diagnósticos
@st.cache_resource
def load_descriptions():
    path = os.path.join(parent_dir, 'insumos', 'descripcion_diagnosticos_traducidos.json')
    with open(path, 'r') as f:
        return json.load(f)

diagnosis_descriptions = load_descriptions()

# ================================
# CARGAR MODELOS
# ================================

# Modelos PyCaret
@st.cache_resource
def load_all_models():
    model_paths = [
        os.path.join(parent_dir, 'modelos', 'modelo_lr'),
        os.path.join(parent_dir, 'modelos', 'modelo_gauss'),
        os.path.join(parent_dir, 'modelos', 'modelo_xgboost'),
        os.path.join(parent_dir, 'modelos', 'modelo_knn'),
        os.path.join(parent_dir, 'modelos', 'modelo_tree')
        ]
    models = [load_model(path) for path in model_paths]
    return models

models = load_all_models()
models_names = ['Regresión', 'Gaussiano', 'XGBoost', 'KNN', 'Árbol',]

# ================================
# INTERFAZ STREAMLIT
# ================================

# Formulario de paciente
with st.form("patient_info"):
    st.header("📝 Datos del paciente")
    patient_name = st.text_input("Nombre completo")
    patient_id = st.text_input("Identificación")
    patient_age = st.number_input("Edad", min_value=0, max_value=120, value=18)
    patient_sex = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro"])
    urgencia = st.selectbox("Nivel de urgencia", ["Alto", "Medio", "Bajo"])
    st.header("💊 Síntomas")
    selected_symptoms_es = st.multiselect(
        "Selecciona los síntomas asociados a tu dolencia:",
        options=list(symptom_translation_rev.keys())
    )
    submitted = st.form_submit_button("🔎 Realizar diagnóstico")

# ================================
# PREDICCIÓN
# ================================

if submitted:
    if not selected_symptoms_es or not patient_name or not patient_age:
        st.warning("⚠️ Por favor completa todos los campos.")
    else:
        selected_symptoms_en = [symptom_translation_rev[s] for s in selected_symptoms_es]
        input_vector = [1 if symptom in selected_symptoms_en else 0 for symptom in symptoms_list]
        input_df = pd.DataFrame([input_vector], columns=symptoms_list)

        predictions = []
        confidences = []

        for i, model in enumerate(models):
            prediction_result = predict_model(model, data=input_df)
            label_column = 'Label' if 'Label' in prediction_result.columns else 'prediction_label'
            score_column = 'Score' if 'Score' in prediction_result.columns else 'prediction_score'

            predicted_label = prediction_result.loc[0, label_column]
            predicted_score = prediction_result.loc[0, score_column]

            predictions.append(predicted_label)
            confidences.append(predicted_score)

        # Mostrar tabla de resultados
        st.subheader("📊 Conclusiones de la junta médica:")
        df_results = pd.DataFrame({
            'Especialista': [f'{i}' for i in models_names],
            'Diagnóstico': [diagnosis_translation.get(p, p) for p in predictions],
            'Confianza': [f"{c*100:.2f}%" for c in confidences],
            'Confianza_num': confidences
        })
        st.dataframe(df_results[['Especialista', 'Diagnóstico', 'Confianza']])

        # Gráfico de barras
        fig = px.bar(df_results, x='Especialista', y='Confianza_num', color='Diagnóstico',
                     text='Confianza', labels={'Confianza_num':'Confianza'}, height=400, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)

        # Votación
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0]

        if most_common[1] >= 3:
            final_diagnosis = most_common[0]
            st.success(f"✅ Diagnóstico final por votación: **{diagnosis_translation.get(final_diagnosis, final_diagnosis)}**")
        else:
            idx_max_conf = np.argmax(confidences)
            final_diagnosis = predictions[idx_max_conf]
            st.success(f"✅ Diagnóstico final por confianza: **{diagnosis_translation.get(final_diagnosis, final_diagnosis)}**")

        # Descripción
        description = diagnosis_descriptions.get(final_diagnosis, "Descripción no disponible.")
        st.info(f"📝 **Descripción:** {description}")

        # Exportar PDF
        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", 'B', 16)
                self.cell(0, 10, "Reporte de Diagnóstico", ln=True, align="C")

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
        pdf.cell(0, 10, f"Diagnóstico: {diagnosis_translation.get(final_diagnosis, final_diagnosis)}", ln=True)
        pdf.multi_cell(0, 10, f"Descripción: {description}")
        pdf.ln(5)
        pdf.cell(0, 10, "Síntomas seleccionados:", ln=True)
        for symptom in selected_symptoms_es:
            pdf.cell(0, 8, f"- {symptom}", ln=True)
        pdf_bytes = bytes(pdf.output(dest='S'))

        st.download_button(
            label="📥 Descargar PDF para historia clínica",
            data=pdf_bytes,
            file_name="diagnostico_paciente.pdf",
            mime="application/pdf"
        )
