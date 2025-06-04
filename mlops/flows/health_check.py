import os
import httpx
from prefect import flow, task, get_run_logger

@task(retries=1, retry_delay_seconds=10)
def ping() -> dict:
    logger = get_run_logger()
    url = os.getenv("API_URL", "http://localhost:8000/health")
    logger.info(f"Llamando a health-check en {url}")
    resp = httpx.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"Respuesta: {data}")
    return data

@flow(name="health-check")
def health_check():
    health = ping()
    logger = get_run_logger()
    logger.info(f"Salud del servicio: {health}")

if __name__ == "__main__":
    health_check()
