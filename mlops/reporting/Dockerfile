FROM python:3.11-slim
WORKDIR /app
COPY requirements-report.txt .
RUN pip install --no-cache-dir -r requirements-report.txt
COPY generate_report.py .
CMD ["python","generate_report.py"]
