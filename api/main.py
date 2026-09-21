import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime

app = FastAPI(title="Energy Forecasting API")

model = joblib.load("model_24h.pkl")


class ForecastRequest(BaseModel):
    target_datetime: datetime
    lag_24h: float = Field(..., gt=0, description="Consumption 24 hours before the target datetime, in MW")
    lag_168h: float = Field(..., gt=0, description="Consumption 168 hours (1 week) before the target datetime, in MW")
    rolling_mean_24h: float = Field(..., gt=0, description="24-hour rolling average consumption leading up to the target datetime, in MW")


class ForecastResponse(BaseModel):
    predicted_mw: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    hour = request.target_datetime.hour
    day_of_week = request.target_datetime.weekday()
    month = request.target_datetime.month

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    features = np.array([[
        hour_sin, hour_cos,
        day_of_week_sin, day_of_week_cos,
        month_sin, month_cos,
        request.lag_24h, request.lag_168h,
        request.rolling_mean_24h
    ]])

    prediction = model.predict(features)[0]

    return ForecastResponse(predicted_mw=float(prediction))