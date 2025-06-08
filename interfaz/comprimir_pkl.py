import pickle
import gzip
from pycaret.classification import load_model, predict_model

modelo = load_model('MODELOS/modelo_discr')

# Guardamos el modelo comprimido
with gzip.open('MODELOS/modelo_discr_comp.pkl.gz', 'wb', compresslevel=9) as f:
    pickle.dump(modelo, f)
