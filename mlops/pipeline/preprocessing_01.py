# pipeline/preprocessing_01.py

import os
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv

#Carga variables  entorno
load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def preprocessing_01(input_path: str = "data/raw/dataset_empresa_03.csv", 
                    output_path: str = "data/processed/dataset_empresa_03_pos_EDA_1.csv") -> str:
    """
    Realiza la primera fase de preprocesamiento de datos:
    1) Carga los datos
    2) Limpia y transforma los datos
    3) Añade características temporales y de ocupación
    4) Registra el dataset resultante en MLflow
    """
    dtype_especifico = {
        "con_clave": "Int64",
        "con_bacoope": "Int64",
        "con_fecha": "str",
        "con_tipo": "string",
        "con_codigo": "string",
        "con_naviera": "string",
        "con_dni": "string",
        "con_tipobillete": "string",
        "con_cupon": "string",
        "con_trayecto": "string",
        "con_shortcomp": "string",
        "con_dateticket": "str",
        "con_acredita": "string",
        "con_nombre": "string",
        "con_intercambiable": "Int64",
        "con_incidencia": "string",
        "buq_nombre": "string",
        "capacidad_pasajeros": "Int64",
        "capacidad_vehiculos": "Int64"
    }
    
    df = pd.read_csv(input_path, dtype=dtype_especifico, low_memory=False)
    
    df["con_fecha"] = pd.to_datetime(df["con_fecha"], errors="coerce")
    df["con_dateticket"] = pd.to_datetime(df["con_dateticket"], errors="coerce")
    
    df.loc[:, "con_desembarcado"] = df["con_incidencia"].fillna("").eq("DESEMBARCADO").astype(int)
    df.drop(columns=["con_incidencia"], inplace=True)
    df = df.query("con_desembarcado != 1")
    
    df.loc[:, "anio_embarque"] = df["con_fecha"].dt.year
    
    df = df[~df["con_tipobillete"].isin(["4", "M"])]
    df.loc[:, "anio"] = df["con_fecha"].dt.year
    
    if 'con_desembarcado' in df.columns:
        df = df[df['con_desembarcado'] != 1]
        df = df.drop(columns=['con_desembarcado'])
   
    df = df.drop_duplicates(subset=['con_codigo', 'con_cupon', 'con_tipobillete'], keep='last')
   
    df.loc[:, "con_fecha"] = pd.to_datetime(df["con_fecha"], errors="coerce")
    df.loc[:, "con_dateticket"] = pd.to_datetime(df["con_dateticket"], errors="coerce")
    df.loc[:, "diferencia_min"] = (df["con_fecha"] - df["con_dateticket"]).dt.total_seconds() / 60
    
    df_copy = df.copy()
   
    df_copy.loc[df_copy['buq_nombre'] == "AL ANDALUS EXPRESS", 'capacidad_pasajeros'] = 215
    df_copy.loc[df_copy['buq_nombre'] == "AL ANDALUS EXPRESS", 'capacidad_vehiculos'] = 150
    df_copy.loc[df_copy['buq_nombre'] == "CECILIA PAYNE", 'capacidad_pasajeros'] = 800
    df_copy.loc[df_copy['buq_nombre'] == "CECILIA PAYNE", 'capacidad_vehiculos'] = 200
    df_copy.loc[df_copy['buq_nombre'] == "HELLENIC HIGHSPEED", 'capacidad_pasajeros'] = 727
    df_copy.loc[df_copy['buq_nombre'] == "HELLENIC HIGHSPEED", 'capacidad_vehiculos'] = 175
    df_copy.loc[df_copy['buq_nombre'] == "JAUME I", 'capacidad_pasajeros'] = 655
    df_copy.loc[df_copy['buq_nombre'] == "JAUME I", 'capacidad_vehiculos'] = 144
    df_copy.loc[df_copy['buq_nombre'] == "KATTEGAT", 'capacidad_pasajeros'] = 974
    df_copy.loc[df_copy['buq_nombre'] == "KATTEGAT", 'capacidad_vehiculos'] = 344
    df_copy.loc[df_copy['buq_nombre'] == "LEVANTE JET", 'capacidad_pasajeros'] = 675
    df_copy.loc[df_copy['buq_nombre'] == "LEVANTE JET", 'capacidad_vehiculos'] = 151
    df_copy.loc[df_copy['buq_nombre'] == "NISSOS CHIOS", 'capacidad_pasajeros'] = 1750
    df_copy.loc[df_copy['buq_nombre'] == "NISSOS CHIOS", 'capacidad_vehiculos'] = 425
    df_copy.loc[df_copy['buq_nombre'] == "PATRIA SEAWAYS", 'capacidad_pasajeros'] = 260
    df_copy.loc[df_copy['buq_nombre'] == "PATRIA SEAWAYS", 'capacidad_vehiculos'] = 480
    df_copy.loc[df_copy['buq_nombre'] == "STENA EUROPE", 'capacidad_pasajeros'] = 1386
    df_copy.loc[df_copy['buq_nombre'] == "STENA EUROPE", 'capacidad_vehiculos'] = 564
    df_copy.loc[df_copy['buq_nombre'] == "TARIFA JET", 'capacidad_pasajeros'] = 800
    df_copy.loc[df_copy['buq_nombre'] == "TARIFA JET", 'capacidad_vehiculos'] = 175
    df_copy.loc[df_copy['buq_nombre'] == "VOLCAN DE TAMASITE", 'capacidad_pasajeros'] = 1500
    df_copy.loc[df_copy['buq_nombre'] == "VOLCAN DE TAMASITE", 'capacidad_vehiculos'] = 300
    df_copy.loc[df_copy['buq_nombre'] == "VOLCAN DE TAUCE", 'capacidad_pasajeros'] = 347
    df_copy.loc[df_copy['buq_nombre'] == "VOLCAN DE TAUCE", 'capacidad_vehiculos'] = 320
    df_copy.loc[df_copy['buq_nombre'] == "WASA EXPRESS", 'capacidad_pasajeros'] = 1500
    df_copy.loc[df_copy['buq_nombre'] == "WASA EXPRESS", 'capacidad_vehiculos'] = 450
    
    tipos_pasajeros = ['1', '2', '3']
    tipos_vehiculos = ['0', '5', '6', '7', '8', '9']
    
    totales = df_copy.groupby('con_bacoope').apply(lambda grupo: pd.Series({
        'total_pasajeros': grupo['con_tipobillete'].isin(tipos_pasajeros).sum(),
        'total_vehiculos': grupo['con_tipobillete'].isin(tipos_vehiculos).sum()
    })).reset_index()
    
    df_copy = pd.merge(df_copy, totales, on='con_bacoope', how='left')
    df_copy['porc_ocupacion_pasajeros'] = (df_copy['total_pasajeros'] / df_copy['capacidad_pasajeros']) * 100
    df_copy['porc_ocupacion_vehiculos'] = (df_copy['total_vehiculos'] / df_copy['capacidad_vehiculos']) * 100
    
    df_copy.loc[:, "con_fecha"] = pd.to_datetime(df_copy["con_fecha"], errors="coerce")
    df_copy.loc[:, "day_of_week"] = df_copy["con_fecha"].dt.dayofweek + 1
    df_copy.loc[:, "is_weekend"] = df_copy["day_of_week"].isin([6, 7]).astype(int)
    df_copy.loc[:, "month"] = df_copy["con_fecha"].dt.month
    df_copy.loc[:, "week_of_year"] = df_copy["con_fecha"].dt.isocalendar().week.astype(int)
    
    conditions = [
        df_copy["month"].isin([12, 1, 2]),   # Invierno
        df_copy["month"].isin([3, 4, 5]),    # Primavera
        df_copy["month"].isin([6, 7, 8]),    # Verano
        df_copy["month"].isin([9, 10, 11])   # Otoño
    ]
    choices = [1, 2, 3, 4]
    df_copy.loc[:, "season"] = np.select(conditions, choices, default=np.nan).astype(int)
    
    #Festivos nacionales
    festivos_nacionales = pd.to_datetime([
        "2022-01-01",  # Año Nuevo
        "2022-01-06",  # Epifanía del Señor (Reyes)
        "2022-04-15",  # Viernes Santo
        "2022-08-15",  # Asunción de la Virgen
        "2022-10-12",  # Fiesta Nacional de España
        "2022-11-01",  # Todos los Santos
        "2022-12-06",  # Día de la Constitución Española
        "2022-12-08",  # Inmaculada Concepción
        "2022-12-25",  # Natividad del Señor (Navidad)
        "2023-01-06",  # Epifanía del Señor (Reyes)
        "2023-04-07",  # Viernes Santo
        "2023-05-01",  # Día del Trabajo
        "2023-08-15",  # Asunción de la Virgen
        "2023-10-12",  # Fiesta Nacional de España
        "2023-11-01",  # Todos los Santos
        "2023-12-06",  # Día de la Constitución Española
        "2023-12-08",  # Inmaculada Concepción
        "2023-12-25",  # Natividad del Señor (Navidad)
        "2024-01-01",  # Año Nuevo
        "2024-01-06",  # Epifanía del Señor (Reyes)
        "2024-03-29",  # Viernes Santo
        "2024-05-01",  # Día del Trabajo
        "2024-08-15",  # Asunción de la Virgen
        "2024-10-12",  # Fiesta Nacional de España
        "2024-11-01",  # Todos los Santos
        "2024-12-06",  # Día de la Constitución Española
        "2024-12-25",  # Natividad del Señor (Navidad)
    ])
    
    #Festivos locales en Algeciras
    festivos_locales = pd.to_datetime([
        "2022-02-28",  # Dia de Andalucia
        "2022-04-14",  # Jueves Santo
        "2022-05-02",  # Lunes siguiente al día del Trabajo
        "2022-06-22",  # Feria Real de Algeciras
        "2022-07-18",  # Festividad de Nuestra Señora del Carmen
        "2022-12-26",  # Lunes siguiente a Navidad
        "2023-01-02",  # Día siguiente a Año Nuevo
        "2023-02-28",  # Día de Andalucía
        "2023-04-06",  # Jueves Santo
        "2023-06-21",  # Miércoles de la Feria Real de Algeciras
        "2023-07-17",  # Lunes posterior a la Festividad de Nuestra Señora del Carmen.​
        "2024-02-28",  # Día de Andalucía
        "2024-03-28",  # Jueves Santo
        "2024-06-26",  # Miércoles de Feria Real de Algeciras
        "2024-07-16",  # Festividad de Nuestra Señora del Carmen
        "2024-12-09",  # Lunes siguiente a la Inmaculada Concepción
    ])
    
    festivos_nacionales_set = set(festivos_nacionales)
    festivos_locales_set = set(festivos_locales)
    
    df_copy.loc[:, "fecha_sin_hora"] = df_copy["con_fecha"].dt.normalize()
    
    df_copy.loc[:, "is_festivo_nacional"] = df_copy["fecha_sin_hora"].isin(festivos_nacionales_set).astype(int)
    df_copy.loc[:, "is_festivo_local"] = df_copy["fecha_sin_hora"].isin(festivos_locales_set).astype(int)
    
    #Festivos especiales - Eid al-Adha
    fechas_eid_aladha = pd.to_datetime([
        "2022-07-09",
        "2023-06-28",
        "2024-06-16"
    ])
    set_eid_aladha = set(fechas_eid_aladha)
    df_copy.loc[:, "is_eid_aladha"] = df_copy["fecha_sin_hora"].isin(set_eid_aladha).astype(int)
    df_copy.loc[:, "is_eid_aladha_prev"] = df_copy["fecha_sin_hora"].apply(
        lambda x: any((x >= fecha - pd.Timedelta(days=7)) and (x < fecha) for fecha in fechas_eid_aladha)
    ).astype(int)
    df_copy.loc[:, "is_eid_aladha_post"] = df_copy["fecha_sin_hora"].apply(
        lambda x: any((x > fecha) and (x <= fecha + pd.Timedelta(days=7)) for fecha in fechas_eid_aladha)
    ).astype(int)
    
    #Festivos especiales - Eid al-Fitr
    fechas_eid_alfitr = pd.to_datetime([
        "2022-05-02",
        "2023-04-21",
        "2024-04-10"
    ])
    set_eid_alfitr = set(fechas_eid_alfitr)
    df_copy.loc[:, "is_eid_alfitr"] = df_copy["fecha_sin_hora"].isin(set_eid_alfitr).astype(int)
    df_copy.loc[:, "is_eid_alfitr_prev"] = df_copy["fecha_sin_hora"].apply(
        lambda x: any((x >= fecha - pd.Timedelta(days=7)) and (x < fecha) for fecha in fechas_eid_alfitr)
    ).astype(int)
    df_copy.loc[:, "is_eid_alfitr_post"] = df_copy["fecha_sin_hora"].apply(
        lambda x: any((x > fecha) and (x <= fecha + pd.Timedelta(days=7)) for fecha in fechas_eid_alfitr)
    ).astype(int)
    
    #Festivos especiales - Mawlid Nabi
    fechas_mawlid_nabi = pd.to_datetime([
        "2022-10-08",
        "2023-09-27",
        "2024-09-15"
    ])
    set_mawlid_nabi = set(fechas_mawlid_nabi)
    df_copy.loc[:, "is_mawlid_nabi"] = df_copy["fecha_sin_hora"].isin(set_mawlid_nabi).astype(int)
    
    df_copy.drop(columns=["fecha_sin_hora"], inplace=True)
    
    df_copy.loc[:, 'con_fecha'] = pd.to_datetime(df_copy['con_fecha'], errors='coerce') 
    df_copy.loc[:, 'hora_embarque'] = df_copy['con_fecha'].dt.hour.astype('int')
    
    df_copy.loc[:, 'agrupacion_billete'] = np.where(df_copy['con_tipobillete'].isin([0, 5, 6, 7, 8, 9]), 0,  
                               np.where(df_copy['con_tipobillete'].isin([1, 2, 3]), 1, 2))
   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_copy.to_csv(output_path, index=False)
    
    return output_path


def run_preprocessing_01(input_path: str = "data/raw/dataset_empresa_03.csv",
                       output_path: str = "data/processed/dataset_empresa_03_pos_EDA_1.csv") -> str:
    """
    Funcion principal para ejecutar el preprocesamiento y registrar en MLflow.
    
    """
    #Configurar MLflow Tracking URI
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    #Iniciar un run para registrar artefactos
    with mlflow.start_run(run_name="Preprocessing_01_Run"):
        processed_path = preprocessing_01(input_path, output_path)
        
        #Leer para obtener información y metricas
        df_processed = pd.read_csv(processed_path)
        
        #Registrar metricas y parametros
        mlflow.log_metric("rows_after_preprocessing", len(df_processed))
        mlflow.log_param("input_dataset", input_path)
        mlflow.log_param("output_dataset", output_path)
        
        #Registrar CSV en MLflow
        mlflow.log_artifact(processed_path, artifact_path="processed_data")
        
    return processed_path

if __name__ == "__main__":
    run_preprocessing_01()