# pipeline/ingestion.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import mlflow
import pymysql  
import logging

#Carga variables entorno
load_dotenv()

def get_connection_params():

    db_host = os.getenv("DB_HOST", "")
    db_user = os.getenv("DB_USER", "")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    logging.info(f"Parámetros de conexión: host={db_host}, db={db_name}, user={db_user}")
    logging.info(f"MLflow URI: {mlflow_uri}")
    
    return {
        "host": db_host,
        "user": db_user,
        "password": db_password,
        "database": db_name,
        "mlflow_uri": mlflow_uri
    }

#Diccionario de capacidades por buque
CAPACIDADES_BUQUES = {
    "AL ANDALUS EXPRESS":   (80,  90),
    "ALCANTARA 2":          (575, 120),
    "ALGECIRAS JET":        (428,  52),
    "AVEMAR DOS":           (855, 174),
    "BAHAMA MAMA":         (1000, 350),
    "BLUE STAR CHIOS":     (1715, 425),
    "BUDA BRIDGE":          (359,  91),
    "CECILIA PAYNE":        (740, 200),
    "CEUTA JET":            (428,  52),
    "CIUDAD DE CEUTA":      (866, 220),
    "CIUDAD DE MALAGA":     (943, 300),
    "DENIA CIUTAT CREATIVA":(399, 430),
    "FORTUNY":             (1250, 140),
    "HELLENIC HIGHSPEED":    (727, 156),
    "JAUME I":               (623, 130),
    "JAUME III":             (623, 130),
    "JJ SISTER":             (806, 220),
    "JUAN J. SISTER":        (806, 220),
    "KATTEGAT":             (1000, 344),
    "LEVANTE JET":           (672, 151),
    "MED STAR":             (1212, 450),
    "MOROCCO STAR":          (935, 225),
    "MOROCCO SUN":          (1001, 280),
    "NAPOLES":              (1600, 481),
    "NISSOS CHIOS":         (1715, 425),
    "PASSIO PER FORMENTERA": (800, 105),
    "PATRIA SEAWAYS":        (277,  60),
    "POETA LOPEZ ANGLADA":  (1257, 243),
    "PONIENTE JET":          (640, 180),
    "REGINA BALTICA":       (1675, 350),
    "SOROLLA":              (1250, 140),
    "STENA EUROPE":         (1220, 310),
    "STENA VINGA":           (400, 200),
    "TANGER EXPRESS":       (900, 344),
    "TARIFA JET":            (900, 200),
    "VILLA DE AGAETE":       (868, 220),
    "VOLCAN DE TAMASITE":   (1469, 403),
    "VOLCAN DE TAUCE":       (347,   0),
    "WASA EXPRESS":          (700, 300)
}


def connect_db():
    params = get_connection_params()
  
    logging.info(f"Conectando a BD en host: {params['host']}, BD: {params['database']}, Usuario: {params['user']}")
    
    uri = f"mysql+pymysql://{params['user']}:{params['password']}@{params['host']}/{params['database']}"
    
    try:
        engine = create_engine(uri, pool_pre_ping=True)
        with engine.connect() as conn:
            logging.info("Conexion a la base de datos establecida correctamente")
        return engine
    except Exception as e:
        logging.error(f"Error al conectar a la base de datos: {str(e)}")
        raise


def generate_dataset(years=(2022, 2023, 2024)) -> pd.DataFrame:
    """
    1) Extrae datos de 'contador'
    2) Extrae mapeo 'barcoope'
    3) Mapea nombres de buque y añade capacidades
    4) Devuelve DataFrame completo
    """
    logging.info(f"Generando dataset para los años: {years}")
    engine = connect_db()

    #EXTRAER datos de contador
    query_cnt = f"""
        SELECT
            con_clave, con_bacoope, con_fecha, con_tipo, con_codigo,
            con_naviera, con_dni, con_tipobillete, con_cupon, con_trayecto,
            con_shortcomp, con_dateticket, con_acredita, con_nombre,
            con_intercambiable, con_incidencia
        FROM contador
        WHERE con_cliente = '03'
          AND YEAR(con_fecha) IN {years}
        ORDER BY con_fecha ASC
    """
    logging.info("Ejecutando consulta principal...")
    df = pd.read_sql(query_cnt, engine)
    logging.info(f"Datos obtenidos: {len(df)} filas")

    #EXTRAER mapeo de buques
    query_map = """
        SELECT bao_codigo, bao_buque
        FROM barcoope
        WHERE bao_cliente = '03'
    """
    logging.info("Obteniendo mapeo de buques...")
    df_map = pd.read_sql(query_map, engine)
    mapping = dict(zip(df_map['bao_codigo'], df_map['bao_buque']))
    logging.info(f"Mapeo de {len(mapping)} buques obtenido")

    #MAPEAR nombres y añadir capacidades
    df['buq_nombre'] = df['con_bacoope'].map(mapping)
    df['capacidad_pasajeros'] = df['buq_nombre'].map(lambda b: CAPACIDADES_BUQUES.get(b, (0,0))[0]).astype(int)
    df['capacidad_vehiculos'] = df['buq_nombre'].map(lambda b: CAPACIDADES_BUQUES.get(b, (0,0))[1]).astype(int)

    return df

def ingest_data(years=(2022, 2023, 2024), output_path: str = "data/raw/dataset_empresa_03.csv") -> str:

    #Obtener la URI de MLflow desde parametros
    mlflow_uri = get_connection_params()["mlflow_uri"]
    logging.info(f"Configurando MLflow con URI: {mlflow_uri}")
    
    #Configurar MLflow Tracking URI
    mlflow.set_tracking_uri(mlflow_uri)

    #Iniciar un run para registrar artefactos
    with mlflow.start_run(run_name="Ingestion_Run"):
        logging.info("Generando dataset...")
        df = generate_dataset(years)

        #Guardar CSV localmente
        logging.info(f"Guardando dataset en {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        #Registrar CSV en MLflow
        logging.info("Registrando artefacto en MLflow...")
        mlflow.log_artifact(output_path, artifact_path="raw_data")
        mlflow.log_param("rows", len(df))
        logging.info(f"Artefacto registrado exitosamente en MLflow ({len(df)} filas)")

    return output_path

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    output = ingest_data()
    print(f"Dataset guardado en: {output}")

