from prefect import flow, task
import subprocess
import os
import sys

@task(retries=2, retry_delay_seconds=60)
def run_report_script():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
  
    script_path = os.path.join(base_dir, "reporting", "generate_report.py")
    
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"El script no existe en la ruta: {script_path}")
    
    
    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script: {e}")
        raise

@flow(name="daily_report")
def daily_report():
    run_report_script()

if __name__ == "__main__":
    daily_report()