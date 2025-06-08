import wikipediaapi
import json
import time
import os
from tqdm import tqdm

# === Parámetros de ejecución ===
INPUT_FILE = 'diagnosticos.txt'
OUTPUT_FILE = 'diagnosis_descriptions.json'
LIMIT_DIAGNOSIS = 400
SLEEP_TIME = 0.1  # segundos entre requests

# === Inicializamos Wikipedia API ===
wiki_en = wikipediaapi.Wikipedia(language = 'en', user_agent='proyectointegrador2/1.0 (gustavojerezt@gmail.com)')

# === Función para obtener resumen desde Wikipedia ===
def get_summary(term):
    # Buscar en inglés
    page_en = wiki_en.page(term)
    if page_en.exists():
        summary = page_en.summary.split('. ')
        short = '. '.join(summary[:2]).strip()
        if not short.endswith('.'):
            short += '.'
        return short

    # No encontrado
    return None

# === Leer la lista de diagnósticos ===
with open(INPUT_FILE, 'r') as f:
    diagnosticos = eval(f.read())

diagnosticos = diagnosticos[:LIMIT_DIAGNOSIS]

# === Cargar progreso previo si existe ===
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        descriptions = json.load(f)
else:
    descriptions = {}

# === Iterar sobre los diagnósticos ===
for diag in tqdm(diagnosticos, desc="Procesando diagnósticos"):
    if diag in descriptions:
        continue  # ya procesado

    try:
        desc = get_summary(diag)
        if desc:
            descriptions[diag] = desc
        else:
            descriptions[diag] = "Descripción no disponible."
    except Exception as e:
        descriptions[diag] = "Error al buscar descripción."
        print(f"Error en diagnóstico: {diag} -> {e}")

    # Guardado parcial tras cada diagnóstico
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    time.sleep(SLEEP_TIME)

print("\n✅ Proceso finalizado. Descripciones almacenadas en:", OUTPUT_FILE)
