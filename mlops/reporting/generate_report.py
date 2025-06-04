#!/usr/bin/env python
# generate_report.py
# -*- coding: utf-8 -*-

import os
import requests
import datetime
import time
import sys
import locale
import codecs
import json
from pathlib import Path


if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: No se pudo importar dotenv. Instalalo con: pip install python-dotenv")
    sys.exit(2)

try:
    from openai import OpenAI
except ImportError:
    print("Error: No se pudo importar OpenAI. Instalalo con: pip install openai")
    sys.exit(2)


try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: No se pudo importar plotly. Instalalo con: pip install plotly")
    sys.exit(2)


load_dotenv()


PROXY_URL = os.getenv("PROXY_URL") or "http://localhost:8000"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-1106-preview")


if not os.getenv("OPENAI_API_KEY"):
    print("Advertencia: OPENAI_API_KEY no está configurada, el resumen LLM fallara")


REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")

PAX_PER_STAFF = 220   # Pasajeros por agente/turno
VEH_PER_STAFF = 55    # Vehículos por agente/turno
PEAK_FACTOR = 1.20    # Umbral día pico (20% sobre media)

def ensure_reports_dir():
    path = Path(REPORTS_DIR)
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Creado directorio de informes: {path}")
    return path


def fetch_predictions(endpoint: str, start_date: str, end_date: str) -> list[int]:

    url = f"{PROXY_URL}{endpoint}"
    payload = {"start_date": start_date, "end_date": end_date}
    for attempt in range(1, 6):
        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("predictions")
            if not isinstance(raw, list):
                raise ValueError(f"Respuesta inesperada: {data!r}")
            preds: list[int] = []
            for item in raw:
                if isinstance(item, dict):
                    for key in ("pasajeros", "vehiculos", "prediction", "value"):
                        if key in item:
                            preds.append(int(item[key]))
                            break
                else:
                    preds.append(int(item))
            return preds
        except requests.RequestException as e:
            print(f"[Intento {attempt}/5] Error al conectar a {url}: {e}")
            time.sleep(5)
    raise ConnectionError(f"No hay conexion a {url} tras varios intentos")


def fetch_prediction_for_date(endpoint: str, date_str: str) -> int:
    preds = fetch_predictions(endpoint, date_str, date_str)
    if len(preds) != 1:
        raise ValueError(f"Se esperaban 1 prediccion para {date_str}, obtenidas: {preds}")
    return preds[0]


def save_report(content: str, filename: str) -> Path:
    reports_dir = ensure_reports_dir()
    file_path = reports_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Informe guardado en: {file_path}")
    return file_path


def save_data_json(data: dict, filename: str) -> Path:
    reports_dir = ensure_reports_dir()
    file_path = reports_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Datos JSON guardados en: {file_path}")
    return file_path


def generate_charts(fechas_str: list[str], pas_preds: list[int], veh_preds: list[int], date_str: str) -> tuple[str, str]:

    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Prediccion de Pasajeros", "Prediccion de Vehículos"),
                        shared_xaxes=True, 
                        vertical_spacing=0.1)
    fig.add_trace(
        go.Scatter(x=fechas_str, y=pas_preds, mode='lines+markers', name='Pasajeros',
                  hovertemplate='<b>Fecha:</b> %{x}<br><b>Pasajeros:</b> %{y:,}<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=fechas_str, y=veh_preds, mode='lines+markers', name='Vehículos',
                  hovertemplate='<b>Fecha:</b> %{x}<br><b>Vehículos:</b> %{y:,}<extra></extra>'),
        row=2, col=1
    )
    fig.update_layout(
        title_text="Predicciones de tráfico para los próximos 15 días",
        height=700,
        hovermode="x unified",
        template="plotly_white"
    )
    fig.update_xaxes(title_text="Fecha", tickangle=45, row=2, col=1)
    fig.update_yaxes(title_text="Pasajeros", row=1, col=1)
    fig.update_yaxes(title_text="Vehículos", row=2, col=1)

    reports_dir = ensure_reports_dir()
    chart_path = reports_dir / f"grafico_{date_str}.html"
    fig.write_html(chart_path, include_plotlyjs='cdn', full_html=True)
    embed_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    return str(chart_path), embed_html


def calc_staff(pax: int, veh: int) -> int:
    return (pax + PAX_PER_STAFF - 1)//PAX_PER_STAFF + (veh + VEH_PER_STAFF - 1)//VEH_PER_STAFF


def main():
    try:
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
        today = datetime.date.today()
        fechas = [today + datetime.timedelta(days=i) for i in range(15)]
        fechas_str = [d.isoformat() for d in fechas]

        print(f"Obteniendo predicciones para: {fechas_str}")
        pas_preds = [fetch_prediction_for_date("/predict/pasajeros", d) for d in fechas_str]
        veh_preds = [fetch_prediction_for_date("/predict/vehiculos", d) for d in fechas_str]

        total_pas = sum(pas_preds)
        total_veh = sum(veh_preds)

        staff_min = [calc_staff(p, v) for p, v in zip(pas_preds, veh_preds)]
        avg_staff = sum(staff_min) // len(staff_min)
        peak_staff = max(staff_min)
        peak_days = [fechas_str[i] for i, s in enumerate(staff_min) if s == peak_staff]

        raw_data = {
            "fecha_generacion": datetime_str,
            "periodo": {"fechas": fechas_str},
            "predicciones": {"pasajeros": pas_preds, "vehiculos": veh_preds},
            "totales": {"pasajeros": total_pas, "vehiculos": total_veh},
            "personal": {
                "minimo_por_dia": staff_min,
                "media": avg_staff,
                "pico": {"valor": peak_staff, "fechas": peak_days}
            }
        }
        save_data_json(raw_data, f"datos_predicciones_{date_str}.json")

        rows = "\n".join(f"| {fechas_str[i]} | {pas_preds[i]} | {veh_preds[i]} |" for i in range(len(fechas_str)))
        report_md = f"""### Boletín diario de demanda (15 días)

Se esperan **{total_pas}** pasajeros y **{total_veh}** vehículos en los próximos 15 días.

| Fecha      | Pasajeros | Vehículos |
|------------|-----------|-----------|
{rows}
"""
        save_report(report_md, f"informe_{date_str}.md")

        print("Generando visualizaciones...")
        chart_path, chart_embed_html = generate_charts(fechas_str, pas_preds, veh_preds, date_str)
        print(f"Gráficos generados en: {chart_path}")

        summary = None
        if os.getenv("OPENAI_API_KEY"):
            client = OpenAI()
            messages = [
                {"role": "system", "content": (
                    "Eres un analista portuario senior con 15 años de experiencia, "
                    "experto en tráfico de pasajeros y vehículos. "
                    "Tu informe debe combinar análisis cuantitativo, identificación de riesgos operativos "
                    "y recomendaciones estratégicas."
                )},
                {"role": "user", "content": f"""{report_md}

**Datos clave:**
- Personal mínimo diario: {staff_min}
- Media quincenal: {avg_staff}
- Pico: {peak_staff} agentes el/los {', '.join(peak_days)}

**Secciones requeridas (usar <h2>):**
1. Resumen de tráfico 
2. Optimización de recursos
3. Riesgos operativos 
4. Recomendaciones 

**Formato:**
- Listas con <ul>/<li>
- Fechas como <strong>24 de mayo</strong>
- Máximo 300 palabras"""}
            ]
            chat = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=600,
                temperature=0.2
            )
            summary = chat.choices[0].message.content.strip()
            full_report = f"{report_md}\n\n{summary}"
            save_report(full_report, f"informe_completo_{date_str}.md")
        else:
            print("Advertencia: No se generó análisis LLM (falta OPENAI_API_KEY)")

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='UTF-8'><title>Informe de predicciones</title>",
            "<style>"
            "body{font-family:Arial;max-width:900px;margin:auto;padding:20px;}"
            "table{width:100%;border-collapse:collapse;}"
            "th,td{border:1px solid #ddd;padding:8px;text-align:right;}"
            "th{background:#f2f2f2;}"
            ".highlight{color:#cc0000;font-weight:bold;}"
            ".charts{margin:30px 0;box-shadow:0 0 10px rgba(0,0,0,0.1);padding:15px;border-radius:5px;}"
            ".table-container{margin:20px 0;overflow-x:auto;}"
            ".dashboard{display:flex;flex-wrap:wrap;gap:20px;margin-bottom:15px;}"
            ".stat-card{flex:1;min-width:200px;background:#f8f8f8;padding:15px;border-radius:8px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1);}"
            ".stat-value{font-size:24px;color:#0066cc;margin:10px 0;}"
            ".stat-label{font-size:14px;color:#555;}"
            ".summary-section{margin:30px 0;box-shadow:0 0 10px rgba(0,0,0,0.1);border-radius:5px;overflow:hidden;}"
            ".summary-section > h2{background:#0066cc;color:white;margin:0;padding:15px;font-size:18px;}"
            ".summary-content{padding:20px;background:#f0f7ff;line-height:1.5;}"
            ".summary-content h2{background:none;color:#333;padding:10px 0;margin:10px 0;font-size:16px;border-bottom:1px solid #ccc;}"
            ".summary-content ul{margin:10px 0;padding-left:25px;}"
            ".summary-content li{margin-bottom:5px;}"
            ".header {display: flex; flex-direction: column; align-items: center; margin-bottom: 2rem; text-align: center;}"
            ".logo {height: 80px; margin-bottom: 1rem;}"
            "</style></head><body>",
            

            "<div class='header'>"
            "<img src='deusto-1.png' alt='Universidad de Deusto' class='logo'>"
            f"<h1>Informe de predicciones {date_str}</h1>"
            "</div>",
            
            f"<p>Generado: {datetime_str}</p>",

            "<div class='dashboard'>",
            f"  <div class='stat-card'><div class='stat-label'>Total pasajeros (15 días)</div><div class='stat-value'>{total_pas:,}</div></div>".replace(",", "."),
            f"  <div class='stat-card'><div class='stat-label'>Total vehículos (15 días)</div><div class='stat-value'>{total_veh:,}</div></div>".replace(",", "."),
            "</div>",
            

            "<div class='dashboard'>",
            f"  <div class='stat-card'><div class='stat-label'>Media diaria pasajeros</div><div class='stat-value'>{total_pas//15:,}</div></div>".replace(",", "."),
            f"  <div class='stat-card'><div class='stat-label'>Media diaria vehículos</div><div class='stat-value'>{total_veh//15:,}</div></div>".replace(",", "."),
            "</div>",
            

            "<div class='dashboard'>",
            f"  <div class='stat-card'><div class='stat-label'>Personal medio / día</div><div class='stat-value'>{avg_staff}</div></div>",
            f"  <div class='stat-card'><div class='stat-label'>Personal pico</div><div class='stat-value'>{peak_staff}</div></div>",
            "</div>",
            

            "<div class='charts'>",
            "<h2>Visualización de predicciones</h2>",
            chart_embed_html,
            "</div>",
            

            "<div class='table-container'>",
            f"<h2>Predicciones del {fechas_str[0]} al {fechas_str[-1]}</h2>",
            "<table><tr><th>Fecha</th><th>Pasajeros</th><th>Vehículos</th><th>Personal recomendado</th></tr>"
        ]
        

        for i, d in enumerate(fechas_str):
            html_parts.append(
                f"<tr><td>{d}</td><td>{pas_preds[i]:,}</td><td>{veh_preds[i]:,}</td><td>{staff_min[i]}</td></tr>"
                .replace(",", ".")
            )
        
        html_parts.append("</table>")
        html_parts.append("</div>")
        

        if summary:
            html_parts.append("""
            <div class='summary-section'>
                <h2>Análisis Inteligente</h2>
                <div class='summary-content'>
            """)
            html_parts.append(summary)
            html_parts.append("</div></div>")
        
        html_parts.append("</body></html>")
        html_content = "\n".join(html_parts)
        save_report(html_content, f"informe_{date_str}.html")

        print("\nProceso completado con exito!")
    except Exception as e:
        print(f"Error en main(): {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()