
# 🩺 Proyecto Integrador 2 
# Maestría de Ciencia de Datos y Analítica - EAFIT - 2025

Este repositorio contiene una aplicación interactiva desarrollada con **Streamlit** y basada en un pipeline de modelación de **PyCaret**, cuyo objetivo es simular una **junta médica** automatizada que, a partir de los síntomas ingresados por un paciente, define un diagnóstico preliminar por consenso entre los cinco mejores modelos de machine learning entrenados sobre el dataset principal.

---

## 📁 Estructura del repositorio

```
proyecto_integrador_2_mcda/
├── .streamlit/
│   └── config.toml               # Configuración del tema de la interfaz Streamlit.
├── datasets/                     # Conjunto de datos de entrenamiento
├── entrenamiento/                # Notebooks de entrenamiento y tunning del pipeline de modelación de PyCaret
├── insumos/                      # Insumos necesarios para el despliegue en Streamlit de la interfaz implementada
├── interfaz/                     # Scripts que implementan y gestionan el despliegue de la interfaz a Streamlit
├── modelos/                      # Objetos PKL de los modelos entrenados.
├── requirements.txt              # Dependencias del proyecto.
```

---

## 🚀 ¿Qué hace esta aplicación?

1. Permite ingresar datos básicos del paciente (nombre, edad, sexo).
2. Ofrece un menú con síntomas comunes para selección múltiple (con autocompletado).
3. Ejecuta predicciones usando cinco modelos previamente entrenados en PyCaret.
4. Aplica una regla de votación para definir el diagnóstico final:
   - ✅ Si **al menos 3 modelos coinciden** en su diagnóstico y **cada uno tiene >60% de confianza**, se adopta esa predicción por **consenso**.
   - 📈 En caso contrario, se adopta el diagnóstico del modelo con **mayor confianza individual**.
5. Muestra una descripción breve del diagnóstico obtenida desde Wikipedia y traducida automáticamente.
6. Permite **exportar un reporte PDF** que incluye los datos del paciente, síntomas seleccionados, resultados individuales y diagnóstico preliminar para la historia clínica.

## 🧠 Tecnologías utilizadas

- [Streamlit](https://streamlit.io) para interfaz de usuario.
- [PyCaret](https://pycaret.org) para entrenamiento y carga de modelos.
- [Plotly](https://plotly.com/python/) para visualización de resultados.
- [ReportLab](https://www.reportlab.com/) o [WeasyPrint](https://weasyprint.org/) para exportación de PDF.
- [Googletrans](https://py-googletrans.readthedocs.io/) para traducción automática.


## 📄 Equipo de desarrollo

### Juan Pablo Bertel Morales - Gustavo Andrés Rubio Castillo - Gustavo Adolfo Jerez Tous 
