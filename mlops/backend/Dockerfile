# ------------------------------------------------------------
#  Dockerfile  —  Imagen del proxy FastAPI (solo CPU)
# ------------------------------------------------------------
    FROM python:3.11-slim

    # (opcional) utilidades de compilación minimas
    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Directorio de trabajo dentro del contenedor
    WORKDIR /app
    
    # 1) Instalar dependencias del proxy
    COPY requirements-serving.txt .
    RUN pip install --no-cache-dir -r requirements-serving.txt
    
    # 2) Copiar el codigo FastAPI
    COPY pipeline/ ./pipeline/
    
    # Variable que serve_fastapi.py utiliza para localizar los artefactos
    ENV MODEL_DIR=/app/models
    
    EXPOSE 8000
    CMD ["uvicorn", "pipeline.serve_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
    
