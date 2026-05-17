from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Churn Prediction API", version="1.0")

MODEL_PATH = "models/churn_xgb.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
else:
    model = None

# All 19 features in the exact order used during training
class CustomerFeatures(BaseModel):
    gender: int              # 0=Female, 1=Male
    SeniorCitizen: int       # 0 or 1
    Partner: int             # 0=No, 1=Yes
    Dependents: int          # 0=No, 1=Yes
    tenure: float
    PhoneService: int        # 0=No, 1=Yes
    MultipleLines: int       # 0=No, 1=No phone, 2=Yes
    InternetService: int     # 0=DSL, 1=Fiber optic, 2=No
    OnlineSecurity: int      # 0=No, 1=No internet, 2=Yes
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int            # 0=Month-to-month, 1=One year, 2=Two year
    PaperlessBilling: int    # 0=No, 1=Yes
    PaymentMethod: int       # 0-3 encoded
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        customer.gender, customer.SeniorCitizen, customer.Partner,
        customer.Dependents, customer.tenure, customer.PhoneService,
        customer.MultipleLines, customer.InternetService, customer.OnlineSecurity,
        customer.OnlineBackup, customer.DeviceProtection, customer.TechSupport,
        customer.StreamingTV, customer.StreamingMovies, customer.Contract,
        customer.PaperlessBilling, customer.PaymentMethod,
        customer.MonthlyCharges, customer.TotalCharges
    ]])

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    risk = "High" if probability >= 0.6 else "Medium" if probability >= 0.3 else "Low"

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk
    )


# ── Forecasting endpoint ───────────────────────────────────────────────────
from fastapi import Query

@app.get("/forecast")
def forecast(
    store: int = Query(default=1, description="Store number 1-10"),
    item:  int = Query(default=1, description="Item number 1-50"),
    days:  int = Query(default=7, description="Days ahead to forecast")
):
    try:
        from use_cases.forecasting.forecast_api import train_and_predict
        result = train_and_predict(
            store=store, item=item, days_ahead=days,
            df_path="data/raw/forecasting/train.csv"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
