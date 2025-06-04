import os
import joblib
import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# URLs de TensorFlow Serving
TF_PAS_URL = os.getenv(
    "TF_PAS_URL",
    "http://localhost:8501/v1/models/pasajeros:predict"
)
TF_VEH_URL = os.getenv(
    "TF_VEH_URL",
    "http://localhost:8502/v1/models/vehiculos:predict"
)

# Artefactos
MODEL_DIR = os.getenv("MODEL_DIR", "models")
lookback = 7
feature_cols_pasajeros: List[str] = []
feature_cols_vehiculos: List[str] = []
scaler_pasajeros = None
scaler_vehiculos = None

# FastAPI init
app = FastAPI(
    title="API Proxy LSTM Ferry",
    version="1.0",
    description="Proxy FastAPI → TensorFlow Serving con pre-/post-procesado"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionInput(BaseModel):
    start_date: str = Field(..., example="2025-05-01")
    end_date: str = Field(..., example="2025-05-07")
    include_features: bool = Field(False)

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Union[str, int, float]]]
    model_info: Dict[str, str]
    features: Optional[Dict[str, Dict[str, float]]] = None

# Funciones de generación de características
def generate_time_features(date: datetime) -> Dict[str, float]:
    features = {}
    day_of_year = date.timetuple().tm_yday
    features['dia_del_anio_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    features['dia_del_anio_cos'] = np.cos(2 * np.pi * day_of_year / 365)
    day = date.day
    features['dia_embarque_sin'] = np.sin(2 * np.pi * day / 31)
    features['dia_embarque_cos'] = np.cos(2 * np.pi * day / 31)
    features['hora_embarque_sin'] = 0
    features['hora_embarque_cos'] = 1
    features['is_weekend'] = 1 if date.weekday() >= 5 else 0
    month = date.month
    features['mes_embarque_sin'] = np.sin(2 * np.pi * month / 12)
    features['mes_embarque_cos'] = np.cos(2 * np.pi * month / 12)
    week = date.isocalendar()[1]
    features['week_of_year_sin'] = np.sin(2 * np.pi * week / 52)
    features['week_of_year_cos'] = np.cos(2 * np.pi * week / 52)
    season_mapping = {12:4,1:4,2:4,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
    season = season_mapping[month]
    for s in range(1,5): features[f'season_{s}'] = 1 if season==s else 0
    weekday = date.weekday()
    for i, name in enumerate(['monday','tuesday','wednesday','thursday','friday','saturday','sunday']):
        features[f'is_{name}'] = 1 if weekday==i else 0
    features['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    features['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    for festivo in ['is_festivo_nacional','is_festivo_local','is_eid_aladha','is_eid_aladha_prev',
                    'is_eid_aladha_post','is_eid_alfitr','is_eid_alfitr_prev','is_eid_alfitr_post','is_mawlid_nabi']:
        features[festivo] = 0
    return features

# Prepara datos para LSTM
def prepare_prediction_data(start_date_str: str, end_date_str: str, target_type: str):
    end = datetime.strptime(end_date_str, '%Y-%m-%d')
    if start_date_str == end_date_str:
        start = end - timedelta(days=lookback-1)
        dates = [start + timedelta(days=i) for i in range(lookback)]
    else:
        start = datetime.strptime(start_date_str, '%Y-%m-%d')
        end = datetime.strptime(end_date_str, '%Y-%m-%d')
        if start > end:
            raise HTTPException(400, "start_date must be <= end_date")
        dates = [start + timedelta(days=i) for i in range((end-start).days+1)]
        if len(dates) < lookback:
            extra = lookback - len(dates)
            new_start = start - timedelta(days=extra)
            dates = [new_start + timedelta(days=i) for i in range(lookback)]
    feats = [generate_time_features(d) for d in dates]
    df = pd.DataFrame(feats)
    cols = feature_cols_pasajeros if target_type=='pasajeros' else feature_cols_vehiculos
    for m in set(cols)-set(df.columns): df[m] = 0
    df = df[cols]
    seq = df.values[-lookback:].astype('float32')
    return np.array([seq]), [dates[-1]]

# Carga scalers y artefactos en startup
def load_artifacts():
    global scaler_pasajeros, scaler_vehiculos, feature_cols_pasajeros, feature_cols_vehiculos, lookback
    scaler_pasajeros = joblib.load(os.path.join(MODEL_DIR,'scaler_pasajeros.pkl'))
    scaler_vehiculos = joblib.load(os.path.join(MODEL_DIR,'scaler_vehiculos.pkl'))
    feature_cols_pasajeros = joblib.load(os.path.join(MODEL_DIR,'feature_columns_pasajeros.pkl'))
    feature_cols_vehiculos = joblib.load(os.path.join(MODEL_DIR,'feature_columns_vehiculos.pkl'))
    import json
    cfg = json.load(open(os.path.join(MODEL_DIR,'config_pasajeros.json')))
    lookback = cfg.get('lookback', lookback)
    logger.info(f"Artifacts loaded: lookback={lookback}")

@app.on_event("startup")
async def startup():
    try:
        load_artifacts()
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")

@app.get("/health")
def health_check():
    return {"status":"ok"}

# Ruta combinada definida antes de la genérica
@app.post("/predict/combined")
def predict_combined(input_data: PredictionInput):
    pas = predict_model('pasajeros', input_data)
    veh = predict_model('vehiculos', input_data)
    veh_map = {d['fecha']: d['vehiculos'] for d in veh.predictions}
    combined = [{
        'fecha': d['fecha'],
        'pasajeros': d['pasajeros'],
        'vehiculos': veh_map.get(d['fecha'], 0)
    } for d in pas.predictions]
    return {
        "predictions": combined,
        "model_info": {
            'pasajeros': pas.model_info,
            'vehiculos': veh.model_info
        },
        "features": {
            'pasajeros': pas.features,
            'vehiculos': veh.features
        }
    }

# Ruta genérica de predicción individual
@app.post("/predict/{model_name}", response_model=PredictionResponse)
def predict_model(model_name: str, input_data: PredictionInput):
    if model_name not in ('pasajeros','vehiculos'):
        raise HTTPException(404, "Model not supported")
    X, dates = prepare_prediction_data(input_data.start_date, input_data.end_date, model_name)
    url = TF_PAS_URL if model_name=='pasajeros' else TF_VEH_URL
    resp = requests.post(url, json={"instances": X.tolist()})
    if resp.status_code != 200:
        raise HTTPException(500, f"TF Serving error: {resp.text}")
    preds_norm = np.array(resp.json().get("predictions", []))
    scaler = scaler_pasajeros if model_name=='pasajeros' else scaler_vehiculos
    preds = scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
    preds = np.round(preds).astype(int)
    results = [
        {"fecha": dates[i].strftime('%Y-%m-%d'), model_name: int(preds[i])}
        for i in range(len(dates))
    ]
    features = None
    if input_data.include_features:
        cols = feature_cols_pasajeros if model_name=='pasajeros' else feature_cols_vehiculos
        features = {
            dates[i].strftime('%Y-%m-%d'): {
                cols[j]: float(X[i][-1][j]) for j in range(len(cols))
            } for i in range(len(dates))
        }
    model_info = {"model_type": "LSTM Bidireccional", "target": model_name, "lookback": str(lookback)}
    return PredictionResponse(predictions=results, model_info=model_info, features=features)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pipeline.serve_fastapi:app", host="0.0.0.0", port=8000, reload=True)