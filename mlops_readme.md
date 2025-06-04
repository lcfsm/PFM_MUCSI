# PFM-MLOPS - Sistema de Predicción de Tráfico de Pasajeros y Vehiculos

## Descripción
Sistema MLOps para predicción de tráfico de pasajeros y vehículos en el puerto de Algeciras utilizando un modelo de DEEP LEARNING (LSTM), con orquestación via Prefect, tracking con MLflow y serving con TensorFlow Serving.

## Inicio Rápido - Arranque Completo del Sistema

### Opción 1: Script Automático (Recomendado)
```powershell
# Ejecutar desde C:\
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run-all.ps1
```

### Opción 2: Arranque Manual por Componentes

#### 1. Servidor MLFlow
```powershell
cd C:\PFM-MLOPS
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root mlflow\mlruns `
  --host 0.0.0.0 `
  --port 5000
```

#### 2. Servidor Prefect
```powershell
cd C:\PFM-MLOPS
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
prefect server start
```

#### 3. Agente Prefect
```powershell
cd C:\PFM-MLOPS
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
prefect worker start --pool "mlops-pool"
```

## Despliegue con Docker

### Construcción y Arranque
```bash
# 1. Construir imagen del proxy (solo primera vez o tras cambios)
docker compose build proxy

# 2. Arrancar toda la solución (proxy + 2 TF-Serving)
docker compose up -d

# 3. Verificar contenedores
docker compose ps
```

### Monitoreo de Logs
```bash
# Logs del proxy
docker compose logs -f proxy

# Logs TF-Serving pasajeros
docker compose logs -f pasajeros

# Logs TF-Serving vehículos
docker compose logs -f vehiculos
```

### Test Rápido
```powershell
# Health check
curl.exe http://localhost:8000/health

# Predicción combinada
curl.exe -X POST http://localhost:8000/predict/combined ^
        -H "Content-Type: application/json" ^
        -d "{\"start_date\":\"2025-05-10\",\"end_date\":\"2025-05-10\",\"include_features\":false}"
```

## Desarrollo y Flujos

### Ejecutar Flujos desde Visual Studio Code
```powershell
PS C:\PFM-MLOPS> .\venv\Scripts\Activate.ps1
(venv) PS C:\PFM-MLOPS> python pipeline\flow_ingesta_entrena.py
```

## Servicios Individuales

### Servidor FastAPI (Uvicorn)
```powershell
cd C:\PFM-MLOPS
.\.venv-serving\Scripts\Activate.ps1
uvicorn backend.pipeline.serve_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

### Servidor Web Frontend (Puerto 8001)
```powershell
cd C:\PFM-MLOPS
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
cd .\frontend\
python -m http.server 8001
```

### TensorFlow Serving - Modelo Pasajeros (Puerto 8501)
```powershell
docker run --rm -p 8501:8501 `
  -v C:/PFM-MLOPS/models/lstm_model_pasajeros_savedmodel:/models/pasajeros/1 `
  -e MODEL_NAME=pasajeros `
  tensorflow/serving
```

### TensorFlow Serving - Modelo Vehículos (Puerto 8502)
```powershell
docker run --rm -p 8502:8501 `
  -v C:/PFM-MLOPS/models/lstm_model_vehiculos_savedmodel:/models/vehiculos/1 `
  -e MODEL_NAME=vehiculos `
  tensorflow/serving
```

## Pruebas de Predicción

### Configurar Payload de Prueba
```powershell
$body = @'
{
  "start_date":"2025-07-11",
  "end_date":"2025-07-11",
  "include_features":false
}
'@
```

### Predicción de Pasajeros
```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/predict/pasajeros `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### Predicción de Vehículos
```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/predict/vehiculos `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### Predicción Combinada
```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/predict/combined `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

## Puertos y Servicios

| Servicio | Puerto | URL |
|----------|--------|-----|
| MLFlow UI | 5000 | http://localhost:5000 |
| Prefect UI | 4200 | http://localhost:4200 |
| FastAPI | 8000 | http://localhost:8000 |
| Frontend | 8001 | http://localhost:8001 |
| TF-Serving Pasajeros | 8501 | http://localhost:8501 |
| TF-Serving Vehículos | 8502 | http://localhost:8502 |

## Notas Importantes

- Tener Docker instalado y funcionando
- Los modelos deben estar entrenados y guardados en la carpeta `models/`
- Para desarrollo, se ha usado el entorno virtual `venv`
- Para serving, se ha usado el entorno virtual `.venv-serving`
- El pool de Prefect debe estar configurado como "mlops-pool"

## Troubleshooting

- Si aparecen problemas con políticas de ejecución en powershell, ejecutar: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
- Para reiniciar completamente Docker: `docker compose down && docker compose up -d`
- Verificar que todos los puertos estén disponibles antes del arranque
