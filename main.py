# telecom_agent/main.py

from fastapi import FastAPI, Query
from model import get_max_kpi, get_anomalies

app = FastAPI(title="Telecom KPI Agent")

@app.get("/max_kpi/")
def max_kpi(
    kpi: str = Query(..., description="KPI to check (e.g., 'SINR')"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    return get_max_kpi(kpi, start_date, end_date)

@app.get("/anomalies/")
def anomalies(
    kpi: str = Query(..., description="KPI to check anomalies for"),
    start_date: str = Query(...),
    end_date: str = Query(...),
    threshold_percentile: int = Query(95)
):
    return get_anomalies(kpi, start_date, end_date, threshold_percentile)
