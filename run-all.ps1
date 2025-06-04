# run-all.ps1

# Ajustes globales
$projectPath = "C:\PFM-MLOPS"
$activateCmd  = ". `"$projectPath\venv\Scripts\Activate.ps1`""

# 1) MLflow Server
Start-Process powershell -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy Bypass",
  "-NoExit",
  "-Command",
  "cd `"$projectPath`"; $activateCmd; mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlflow\mlruns --host 0.0.0.0 --port 5000"
)

# 2) Prefect Server
Start-Process powershell -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy Bypass",
  "-NoExit",
  "-Command",
  "cd `"$projectPath`"; $activateCmd; prefect server start"
)

# 3) Prefect Worker
Start-Process powershell -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy Bypass",
  "-NoExit",
  "-Command",
  "cd `"$projectPath`"; $activateCmd; prefect worker start --pool `"mlops-pool`""
)
