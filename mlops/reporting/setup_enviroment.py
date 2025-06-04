#!/usr/bin/env python
"""
Script para configurar el entorno de reporting
Verifica la estructura de directorios y las dependencias necesarias
"""

import os
import sys
import subprocess
import platform

def ensure_dir(dir_path):
    """Crea un directorio si no existe"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Creado directorio: {dir_path}")
    else:
        print(f"El directorio ya existe: {dir_path}")

def check_and_install_dependencies():
    """Verifica e instala las dependencias necesarias"""
    dependencies = [
        "python-dotenv",
        "openai",
        "requests",
        "prefect"
    ]
    
    print("Verificando dependencias...")
    
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✓ {dep} ya está instalado")
        except ImportError:
            print(f"✗ Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

def create_sample_env():
    """Crea un archivo .env de ejemplo si no existe"""
    env_path = os.path.join(os.getcwd(), ".env")
    
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("""# Variables de entorno para el sistema de reporting
PROXY_URL=http://localhost:8000
OPENAI_API_KEY=your_openai_api_key_here
""")
        print(f"Creado archivo .env de ejemplo en {env_path}")
    else:
        print(f"El archivo .env ya existe en {env_path}")

def setup_project_structure():
    """Configura la estructura del proyecto"""
    base_dir = os.getcwd()
    

    reporting_dir = os.path.join(base_dir, "reporting")
    ensure_dir(reporting_dir)
    
 
    files_to_create = {
        "reporting/generate_report.py": """#!/usr/bin/env python
# generate_report.py - Generador de reportes diarios
# Este archivo debe estar en la carpeta reporting/

import os
import requests
import datetime
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Lee la URL del proxy de la variable de entorno
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:8000")

# El resto del código sigue igual...
""",
        "reporting/reporting_flow.py": """from prefect import flow, task
import subprocess
import os
import sys

@task(retries=2, retry_delay_seconds=60)
def run_report_script():
    # Obtener la ruta del script relativa a este archivo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "generate_report.py")
    
    # Ejecutar el script con el entorno actual
    subprocess.run(
        [sys.executable, script_path],
        check=True
    )

@flow(name="daily_report")
def daily_report():
    run_report_script()

if __name__ == "__main__":
    daily_report()
"""
    }
    
    for file_path, content in files_to_create.items():
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
       
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                f.write(content)
            print(f"Creado archivo: {file_path}")
        else:
            print(f"El archivo ya existe: {file_path}")

def main():
    print("=== Configurando entorno de reporting ===")
    
 
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    
    print(f"Sistema operativo: {platform.system()} {platform.release()}")
    
   
    setup_project_structure()
    
    
    check_and_install_dependencies()
    
    
    create_sample_env()
    
    print("\n=== Configuración completada ===")
    print("""
Para ejecutar el flujo de reporting:
1. Edita el archivo .env y establece las variables necesarias
2. Ejecuta el flujo con: python -m reporting.reporting_flow
   o desde Prefect con: prefect deployment run daily_report
""")

if __name__ == "__main__":
    main()