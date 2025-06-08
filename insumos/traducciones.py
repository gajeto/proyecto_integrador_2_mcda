import json
from deep_translator import GoogleTranslator

# Leemos los síntomas
with open('lista_sintomas.txt', 'r') as f:
    symptoms_list = eval(f.read())

# Leemos los diagnósticos (solo primeros 400)
with open('diagnosticos.txt', 'r') as f:
    diagnosis_list = eval(f.read())
#diagnosis_list = diagnosis_list[:400]

# Leemos las descripciones originales de Wikipedia (en inglés)
with open('diagnosis_descriptions.json', 'r') as f:
    diagnosis_descriptions = json.load(f)

# ==========================================
# Función general de traducción controlada
# ==========================================

def translate_list(items, src='en', dest='es'):
    translation_dict = {}
    for item in items:
        try:
            translated = GoogleTranslator(source=src, target=dest).translate(item)
            translation_dict[item] = translated.capitalize()
        except Exception as e:
            print(f"Error al traducir '{item}': {e}")
            translation_dict[item] = item
    return translation_dict

# ==========================================
# Traducción de síntomas y diagnósticos
# ==========================================

print("Traduciendo síntomas...")
#symptom_translation = translate_list(symptoms_list)

print("Traduciendo diagnósticos...")
diagnosis_translation = translate_list(diagnosis_list)

# ==========================================
# Traducción de descripciones de Wikipedia
# ==========================================
'''
print("Traduciendo descripciones de diagnósticos...")
diagnosis_descriptions_es = {}

for diag_key, description_en in diagnosis_descriptions.items():
    try:
        translated_desc = GoogleTranslator(source='en', target='es').translate(description_en)
        diagnosis_descriptions_es[diag_key] = translated_desc
    except Exception as e:
        print(f"Error al traducir descripción de '{diag_key}': {e}")
        diagnosis_descriptions_es[diag_key] = description_en  # fallback al inglés

# ==========================================
# Guardado de archivos finales
# ==========================================

with open('symptom_translation.json', 'w') as f:
    json.dump(symptom_translation, f, indent=2, ensure_ascii=False)
'''
with open('diagnosis_translation.json', 'w') as f:
    json.dump(diagnosis_translation, f, indent=2, ensure_ascii=False)

#with open('diagnosis_descriptions_es.json', 'w') as f:
#    json.dump(diagnosis_descriptions_es, f, indent=2, ensure_ascii=False)

print("✅ Traducción completa guardada.")
