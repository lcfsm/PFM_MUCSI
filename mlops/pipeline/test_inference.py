#!/usr/bin/env python
"""
Script para probar el servicio de inferencia FastAPI.
"""
import requests
import json
import argparse
from datetime import datetime, timedelta
import logging

#Configuración logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Probar el servicio de inferencia FastAPI")
    parser.add_argument("--url", default="http://localhost:8000", help="URL del servicio")
    parser.add_argument("--type", choices=["pasajeros", "vehiculos", "combined"], default="combined", 
                        help="Tipo de predicción a realizar")
    parser.add_argument("--days", type=int, default=30, help="Número de días a predecir")
    
    args = parser.parse_args()
   
    today = datetime.now()
    start_date = today.strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    request_data = {
        "start_date": start_date,
        "end_date": end_date,
        "include_features": True
    }
    
    endpoint = f"/predict/{args.type}"
    url = f"{args.url}{endpoint}"
    
    logger.info(f"Enviando solicitud a {url}")
    logger.info(f"Datos: {json.dumps(request_data)}")
    
    try:
        response = requests.post(url, json=request_data)
        response.raise_for_status()
     
        data = response.json()
        logger.info("\n=== Resultados de la predicción ===")
        logger.info(f"Status code: {response.status_code}")
        
        print("\nPredicciones:")
        for pred in data["predictions"][:5]:  
            print(f"  - {pred}")
        
        if len(data["predictions"]) > 5:
            print(f"    ... y {len(data['predictions']) - 5} más")
            
        print("\nInformación del modelo:")
        for key, value in data["model_info"].items():
            print(f"  - {key}: {value}")
            
        logger.info("Prueba completada con éxito!")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error en la solicitud HTTP: {e}")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()