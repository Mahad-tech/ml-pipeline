import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.ingestion.loader import load_csv
from src.features.transformer import transform_churn

HIGH_RISK_CUSTOMER = {
    "gender": 1, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
    "tenure": 2, "PhoneService": 1, "MultipleLines": 0,
    "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
    "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
    "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
    "PaymentMethod": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0
}

LOW_RISK_CUSTOMER = {
    "gender": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 1,
    "tenure": 60, "PhoneService": 1, "MultipleLines": 1,
    "InternetService": 0, "OnlineSecurity": 2, "OnlineBackup": 2,
    "DeviceProtection": 2, "TechSupport": 2, "StreamingTV": 0,
    "StreamingMovies": 0, "Contract": 2, "PaperlessBilling": 0,
    "PaymentMethod": 0, "MonthlyCharges": 45.0, "TotalCharges": 2700.0
}

def test_load_csv():
    df = load_csv("data/raw/churn/churn.csv")
    assert df.shape[0] == 7043
    assert df.shape[1] == 21

def test_transform_churn_shape():
    df = load_csv("data/raw/churn/churn.csv")
    X, y = transform_churn(df)
    assert X.shape[1] == 19
    assert len(y) == 7043

def test_no_nulls_after_transform():
    df = load_csv("data/raw/churn/churn.csv")
    X, y = transform_churn(df)
    assert X.isnull().sum().sum() == 0

def test_target_is_binary():
    df = load_csv("data/raw/churn/churn.csv")
    _, y = transform_churn(df)
    assert set(y.unique()).issubset({0, 1})

def test_api_health():
    from fastapi.testclient import TestClient
    from src.serving.api import app
    client = TestClient(app)
    assert client.get("/health").status_code == 200

def test_api_prediction_high_risk():
    from fastapi.testclient import TestClient
    from src.serving.api import app
    client = TestClient(app)
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert data["risk_level"] != "Low"  # high-risk customer should not be Low

def test_api_prediction_low_risk():
    from fastapi.testclient import TestClient
    from src.serving.api import app
    client = TestClient(app)
    response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "Low"  # loyal long-term customer
