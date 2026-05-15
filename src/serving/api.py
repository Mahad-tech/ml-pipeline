from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Churn Prediction API", version="1.0")

MODEL_PATH = "models/churn_xgb.pkl"

# Load model at startup
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
else:
    model = None
    print("WARNING: No model found. Run training first.")

class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: int        # 0=Month-to-month, 1=One year, 2=Two year
    InternetService: int # 0=DSL, 1=Fiber optic, 2=No

class PredictionResponse(BaseModel):
    churn_prediction: int       # 0 or 1
    churn_probability: float    # 0.0 to 1.0
    risk_level: str             # Low / Medium / High

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
        customer.tenure,
        customer.MonthlyCharges,
        customer.TotalCharges,
        customer.Contract,
        customer.InternetService
    ]])

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    if probability < 0.3:
        risk = "Low"
    elif probability < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk
    )
