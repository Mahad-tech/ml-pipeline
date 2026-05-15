import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
from src.ingestion.loader import load_csv
from src.features.transformer import transform_churn

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
    response = client.get("/health")
    assert response.status_code == 200

def test_api_prediction_returns_risk():
    from fastapi.testclient import TestClient
    from src.serving.api import app
    client = TestClient(app)
    payload = {
        "tenure": 2,
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
        "Contract": 0,
        "InternetService": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]
