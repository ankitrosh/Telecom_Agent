import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


# Load data and model
df = pd.read_csv("data/AD_data_10KPI.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Load trained autoencoder
model = load_model("data/autoencoder_model.h5", compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError()) 
kpi_cols = [col for col in df.columns if col not in ['Site_ID', 'Sector_ID', 'Date']]

def get_max_kpi(kpi: str, start_date: str, end_date: str):
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if kpi not in df_filtered.columns:
        return {"error": f"Invalid KPI: {kpi}"}
    
    if df_filtered.empty:
        return {"error": "No data in the given date range."}
    
    if df_filtered[kpi].isna().all():
        return {"error": f"All values for KPI '{kpi}' are missing in the selected range."}

    idx = df_filtered[kpi].idxmax()
    row = df_filtered.loc[idx]
    return row.to_dict()


def get_anomalies(kpi: str, start_date: str, end_date: str, threshold_percentile=95):
    X = df[kpi_cols].values
    X_pred = model.predict(X)
    reconstruction_error = np.mean((X - X_pred) ** 2, axis=1)
    df['reconstruction_error'] = reconstruction_error
    threshold = np.percentile(reconstruction_error, threshold_percentile)
    df['anomaly'] = reconstruction_error > threshold

    filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['anomaly'] == True)]
    return filtered[['Date', 'Site_ID', 'Sector_ID', kpi, 'reconstruction_error']].to_dict(orient="records")
