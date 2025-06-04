#Asi esta funcionando correctamente desde Prefect

import sys
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from prefect import flow, task, get_run_logger
import mlflow
import os
from ingestion import ingest_data
from preprocessing_01 import run_preprocessing_01
from preprocessing_02 import run_preprocessing_02
from datetime import datetime
from dotenv import load_dotenv
import importlib.util

from pathlib import Path

#Primero cargar desde .env (modo desarrollo/VS)
load_dotenv()

#Luego configurar integracion con secrets de Prefect
import logging

#Configuracion de logging basico para cuando estamos fuera del contexto de Prefect
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("flow_script")

def get_env_or_secret(key, default=None):
    #Primero intentar obtener desde variable de entorno (ya cargada por load_dotenv)
    value = os.getenv(key)
    
    #Si no existe y estamos en un contexto de Prefect, intentar obtener el secreto
    if not value and "PREFECT__FLOW_RUN_ID" in os.environ:
        try:
            from prefect.blocks.system import Secret
            #Mapear nombres de variables a nombres de secretos
            secret_names = {
                "MLFLOW_TRACKING_URI": "mlflow-tracking-uri",
                "DB_HOST": "db-host",
                "DB_USER": "db-user",
                "DB_PASSWORD": "db-password",
                "DB_NAME": "db-name"
            }
            
            secret_name = secret_names.get(key)
            if secret_name:
                secret_block = Secret.load(secret_name)
                value = secret_block.get()
                logger.info(f"Variable {key} cargada desde secret {secret_name}")
        except Exception as e:
            logger.warning(f"No se pudo cargar el secret para {key}: {str(e)}")
    
    return value or default

#Configurar MLflow con la URI desde variables de entorno o secrets
MLFLOW_TRACKING_URI = get_env_or_secret("MLFLOW_TRACKING_URI", "http://localhost:5000")
logger.info(f"MLFLOW_TRACKING_URI configurado como: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#Exportar para que los modulos importados tengan acceso
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["DB_HOST"] = get_env_or_secret("DB_HOST", "tme.ccdtbj7azle2.eu-west-1.rds.amazonaws.com")
os.environ["DB_USER"] = get_env_or_secret("DB_USER", "mastertools")
os.environ["DB_PASSWORD"] = get_env_or_secret("DB_PASSWORD", "CriurnrFKF66jt")
os.environ["DB_NAME"] = get_env_or_secret("DB_NAME", "mthydroholding")
logger.info(f"Variables de entorno configuradas: DB_HOST={os.environ['DB_HOST']}, DB_NAME={os.environ['DB_NAME']}")


def import_train_function():
    try:
        #Intentar usar el logger de Prefect si esta en contexto de flujo
        try:
            task_logger = get_run_logger()
        except:
            task_logger = logger
        
        module_path = Path(__file__).parent / "lstm_module_temp.py"
        original_path = Path(__file__).parent / "(pv)lstm_training.py"
        
        task_logger.info(f"Verificando rutas de modulos: {module_path}, {original_path}")
        
        if not module_path.exists():
            #Crear un enlace temporal 
            task_logger.info(f"Creando copia temporal de modulo de entrenamiento")
            try:
                if not original_path.exists():
                    task_logger.error(f"Archivo original no encontrado: {original_path}")
                    raise FileNotFoundError(f"No se encontro el archivo {original_path}")
                    
                module_path.write_text(original_path.read_text(encoding='utf-8'), encoding='utf-8')
                task_logger.info(f"Modulo temporal creado correctamente: {module_path}")
            except Exception as e:
                task_logger.error(f"Error al crear modulo temporal: {str(e)}")
                raise

        spec = importlib.util.spec_from_file_location("lstm_training", str(module_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["lstm_training"] = module
        spec.loader.exec_module(module)
        task_logger.info("Modulo de entrenamiento importado correctamente")
        return module.run_train_lstm
    except Exception as e:
        logger.error(f"Error al importar modulo de entrenamiento: {str(e)}")
        raise

@task(retries=3, retry_delay_seconds=60)
def task_ingest():
    task_logger = get_run_logger()
    task_logger.info("Iniciando ingesta de datos desde la base de datos...")
    task_logger.info(f"Usando MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    task_logger.info(f"Usando DB_HOST: {os.getenv('DB_HOST')}")
    task_logger.info(f"Usando DB_USER: {os.getenv('DB_USER')}")
    task_logger.info(f"Usando DB_NAME: {os.getenv('DB_NAME')}")
    
    try:
        ingest_data()
        task_logger.info("Ingesta finalizada. Dataset original generado.")
    except Exception as e:
        task_logger.error(f"Error durante la ingesta: {str(e)}")
        #Mostrar traceback completo para diagnostico
        import traceback
        task_logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        raise

@task(retries=2, retry_delay_seconds=30)
def task_preprocessing_01():
    logger = get_run_logger()
    logger.info("Ejecutando preprocesamiento 01...")
    run_preprocessing_01()
    logger.info("Preprocesamiento 01 finalizado.")

@task(retries=2, retry_delay_seconds=30)
def task_preprocessing_02():
    logger = get_run_logger()
    logger.info("Ejecutando preprocesamiento 02...")
    run_preprocessing_02()
    logger.info("Preprocesamiento 02 finalizado.")

@task
def task_train_model(run_name: str = None):
    logger = get_run_logger()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if run_name is None else run_name

    logger.info("Iniciando entrenamiento LSTM...")

    try:
        train_lstm_model = import_train_function()
        train_lstm_model()
        logger.info("Entrenamiento finalizado. Artefactos y metricas registrados en MLflow.")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise

@flow(name="Flow_Ingesta_Entrena_LSTM")
def main_flow():
    flow_logger = get_run_logger()
    flow_logger.info("Iniciando flow principal...")

    flow_logger.info(f"DB_HOST: {os.getenv('DB_HOST')}")
    flow_logger.info(f"DB_USER: {os.getenv('DB_USER')}")
    flow_logger.info(f"DB_NAME: {os.getenv('DB_NAME')}")
    flow_logger.info(f"MLflow configurado en: {os.getenv('MLFLOW_TRACKING_URI')}")
 
    if not os.getenv("DB_HOST"):
        os.environ["DB_HOST"] = "tme.ccdtbj7azle2.eu-west-1.rds.amazonaws.com"
        flow_logger.info(f"Usando DB_HOST por defecto: {os.environ['DB_HOST']}")
    
    if not os.getenv("DB_USER"):
        os.environ["DB_USER"] = "mastertools"
        flow_logger.info(f"Usando DB_USER por defecto: {os.environ['DB_USER']}")
        
    if not os.getenv("DB_PASSWORD"):
        os.environ["DB_PASSWORD"] = "CriurnrFKF66jt"
        flow_logger.info("Usando DB_PASSWORD por defecto (no mostrada por seguridad)")
        
    if not os.getenv("DB_NAME"):
        os.environ["DB_NAME"] = "mthydroholding"
        flow_logger.info(f"Usando DB_NAME por defecto: {os.environ['DB_NAME']}")
    
    task_ingest()
    task_preprocessing_01()
    task_preprocessing_02()
    task_train_model()
    
    flow_logger.info("Flow completado exitosamente.")

if __name__ == "__main__":
    main_flow()