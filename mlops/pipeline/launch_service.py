#!/usr/bin/env python
"""
Script para lanzar el servicio de inferencia FastAPI.
"""
import os
import argparse
import subprocess
import logging

#Configuracion de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def check_requirements():
    try:
        import fastapi
        import uvicorn
        logger.info("FastAPI y Uvicorn estan instalados.")
        return True
    except ImportError:
        logger.error("FastAPI o Uvicorn no estan instalados. Instalando...")
        try:
            subprocess.run(["pip", "install", "fastapi", "uvicorn[standard]"], check=True)
            logger.info("FastAPI y Uvicorn instalados correctamente.")
            return True
        except subprocess.CalledProcessError:
            logger.error("Error al instalar las dependencias.")
            return False

def main():
    parser = argparse.ArgumentParser(description="Lanzar el servicio de inferencia FastAPI")
    parser.add_argument("--host", default="0.0.0.0", help="Host para ejecutar el servidor")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para ejecutar el servidor")
    parser.add_argument("--reload", action="store_true", help="Activar recarga automatica en desarrollo")
    parser.add_argument("--workers", type=int, default=1, help="Numero de workers para Uvicorn")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"], 
                        help="Nivel de log")
    
    args = parser.parse_args()
    
    if not check_requirements():
        logger.error("No se pudieron instalar las dependencias necesarias. Abortando.")
        return
    
    #Construir comando para Uvicorn
    cmd = [
        "uvicorn",
        "serve_fastapi:app",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--log-level={args.log_level}",
        f"--workers={args.workers}"
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    logger.info(f"Iniciando servidor con comando: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Servidor detenido por el usuario.")
    except Exception as e:
        logger.error(f"Error al iniciar el servidor: {str(e)}")

if __name__ == "__main__":
    main()