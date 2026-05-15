from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np

app = FastAPI(title="Churn Prediction API")

class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: int        # 0=Month-to-month, 1=One year, 2=Two year
    InternetService: int # 0=DSL, 1=Fiber, 2=No

@app.get("/")
def root():
    return {"status": "ok", "model": "churn-logistic-regression"}

@app.post("/predict")
def predict(customer: CustomerFeatures):
    features = np.array([[
        customer.tenure,
        customer.MonthlyCharges,
        customer.TotalCharges,
        customer.Contract,
        customer.InternetService
    ]])
    # model would be loaded from mlflow or pickle in production
    return {"message": "Model not loaded yet — wire up after training"}
