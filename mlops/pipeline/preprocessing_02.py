# pipeline/preprocessing_02.py

import os
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv

load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def preprocessing_02(
    input_path: str = "data/processed/dataset_empresa_03_pos_EDA_1.csv",
    output_path: str = "data/processed/dataset_empresa_03_pos_EDA_2.csv"
) -> str:
    """
    Realiza la segunda fase de EDA y preparacion de datos:
    1) Carga el dataset
    2) Conversion de fechas y extracción de años
    3) Clasificacion de tipo de billete
    4) Generación de variables temporales
    5) Preparacion de subsets (festivos, rutas, propiedades)
    6) Seleccion y transformacion de variables para el modelo
    7) Creacion de variables sinusoidales y one-hot encoding
    8) Guardado del dataset final
    """
    df = pd.read_csv(input_path, low_memory=False)

    for col in ["con_fecha", "con_dateticket"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["anio_con_fecha"] = df["con_fecha"].dt.year
    df["anio_con_dateticket"] = df["con_dateticket"].dt.year

    def clasificar_billete(codigo):
        if codigo in [1, 2, 3]:
            return "Pasajero"
        elif codigo in [0, 5, 6, 7, 8, 9]:
            return "Vehículo"
        else:
            return "Otro"

    df["tipo_billete_clasificado"] = df["con_tipobillete"].apply(clasificar_billete)

    df["fecha_mensual"] = df["con_fecha"].dt.to_period("M").dt.to_timestamp()
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    df["nombre_dia"] = df["day_of_week"].map(dict(zip(range(7), dias_semana)))

    df_festivo = df.copy()
    df_festivo["tipo_billete_clasificado"] = df_festivo["con_tipobillete"].apply(clasificar_billete)

    def tipo_festivo(row):
        if row.get("is_eid_aladha", 0) == 1:
            return "Eid al-Adha"
        elif row.get("is_eid_alfitr", 0) == 1:
            return "Eid al-Fitr"
        elif row.get("is_mawlid_nabi", 0) == 1:
            return "Mawlid"
        elif row.get("is_festivo_nacional", 0) == 1:
            return "Festivo Nacional"
        elif row.get("is_festivo_local", 0) == 1:
            return "Festivo Local"
        else:
            return None

    df_festivo["tipo_festivo"] = df_festivo.apply(tipo_festivo, axis=1)

    df_rutas = df.copy()
    df_rutas["tipo_billete_clasificado"] = df_rutas["con_tipobillete"].apply(clasificar_billete)

    df_prop = df.copy()
    def agrupar_billete(codigo):
        if codigo in [1, 2, 3]:
            return "Pasajero"
        elif codigo in [0, 5, 6, 7, 8, 9]:
            return "Vehículo"
        else:
            return "Otro"

    df_prop["categoria_billete"] = df_prop["con_tipobillete"].apply(agrupar_billete)
    df_prop["con_fecha"] = pd.to_datetime(df_prop["con_fecha"], errors="coerce")
    df_prop["fecha_mensual"] = df_prop["con_fecha"].dt.to_period("M").dt.to_timestamp()

    df["fecha_sin_hora"] = df["con_fecha"].dt.date
    df["categoria"] = df["con_tipobillete"].apply(clasificar_billete)
    df["anio"] = df["con_fecha"].dt.year
    df["hora"] = df["con_fecha"].dt.hour

    columnas_a_excluir = [
        "con_clave", "con_bacoope", "con_tipo", "con_acredita", "con_nombre",
        "con_codigo", "con_naviera", "con_dni", "con_cupon", "con_trayecto",
        "con_shortcomp", "buq_nombre", "anio", "anio_con_fecha", "anio_con_dateticket",
        "agrupacion_billete", "fecha_mensual", "hora", "nombre_dia",
        "tipo_billete_clasificado"
    ]
    df_modelo = df.drop(columns=columnas_a_excluir).copy()
    df_modelo = df_modelo.rename(
        columns={"fecha_sin_hora": "fecha_embarque", "categoria": "agrupacion"}
    )

    df_modelo["con_fecha"] = pd.to_datetime(df_modelo["con_fecha"], errors="coerce")
    df_modelo["dia_embarque"] = (df_modelo["con_fecha"].dt.dayofweek + 1).astype("int64")
    df_modelo["tipo_agrupacion"] = df_modelo["agrupacion"].apply(lambda x: 1 if x == "Pasajero" else 0)
    df_modelo.drop(columns=["agrupacion"], inplace=True)

    df_modelo["mes_embarque"] = df_modelo["con_fecha"].dt.month
    df_modelo.drop(columns=["day_of_week", "month"], inplace=True)
    df_modelo["fecha_embarque"] = pd.to_datetime(df_modelo["fecha_embarque"], errors="coerce")

    df_modelo = pd.get_dummies(df_modelo, columns=["anio_embarque"], prefix="anio")
    one_hot_cols = [col for col in df_modelo.columns if col.startswith("anio_")]
    df_modelo[one_hot_cols] = df_modelo[one_hot_cols].astype(int)

    df_modelo["dia_del_anio"] = df_modelo["fecha_embarque"].dt.dayofyear
    one_hot_season = pd.get_dummies(df_modelo["season"], prefix="season").astype(int)
    df_modelo = pd.concat([df_modelo, one_hot_season], axis=1)

    df_final = df_modelo.copy()
    df_final["dia_del_anio_sin"] = np.sin(2 * np.pi * df_final["dia_del_anio"] / 366)
    df_final["dia_del_anio_cos"] = np.cos(2 * np.pi * df_final["dia_del_anio"] / 366)
    df_final["dia_embarque_sin"] = np.sin(2 * np.pi * df_final["dia_embarque"] / 7)
    df_final["dia_embarque_cos"] = np.cos(2 * np.pi * df_final["dia_embarque"] / 7)
    df_final["hora_embarque_sin"] = np.sin(2 * np.pi * df_final["hora_embarque"] / 24)
    df_final["hora_embarque_cos"] = np.cos(2 * np.pi * df_final["hora_embarque"] / 24)
    df_final["mes_embarque_sin"] = np.sin(2 * np.pi * df_final["mes_embarque"] / 12)
    df_final["mes_embarque_cos"] = np.cos(2 * np.pi * df_final["mes_embarque"] / 12)
    df_final["week_of_year_sin"] = np.sin(2 * np.pi * df_final["week_of_year"] / 53)
    df_final["week_of_year_cos"] = np.cos(2 * np.pi * df_final["week_of_year"] / 53)

    df_final["weekday"] = df_final["con_fecha"].dt.weekday
    day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, name in enumerate(day_names):
        df_final[f"is_{name}"] = (df_final["weekday"] == i).astype(int)
    df_final.drop(columns=["weekday"], inplace=True)

    df_final["weekday"] = df_final["con_fecha"].dt.weekday
    df_final["weekday_sin"] = np.sin(2 * np.pi * df_final["weekday"] / 7)
    df_final["weekday_cos"] = np.cos(2 * np.pi * df_final["weekday"] / 7)
    for i, name in enumerate(day_names):
        df_final[f"is_{name}"] = (df_final["weekday"] == i).astype(int)
    df_final.drop(columns=["weekday"], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)

    return output_path

def run_preprocessing_02(
    input_path: str = "data/processed/dataset_empresa_03_pos_EDA_1.csv",
    output_path: str = "data/processed/dataset_empresa_03_pos_EDA_2.csv"
) -> str:
    """
    Ejecuta preprocessing_02 con MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="Preprocessing_02_Run"):
        result = preprocessing_02(input_path, output_path)
        df_res = pd.read_csv(result)
        mlflow.log_metric("rows_after_preprocessing_02", len(df_res))
        mlflow.log_param("input_dataset", input_path)
        mlflow.log_param("output_dataset", output_path)
        mlflow.log_artifact(result, artifact_path="processed_data_02")
    return result

if __name__ == "__main__":
    run_preprocessing_02()
