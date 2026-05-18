import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

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

# --- Unit tests (no data or model files needed) ---

def test_feature_count():
    assert len(HIGH_RISK_CUSTOMER) == 19
    assert len(LOW_RISK_CUSTOMER) == 19

def test_contract_values_are_valid():
    assert HIGH_RISK_CUSTOMER["Contract"] in [0, 1, 2]
    assert LOW_RISK_CUSTOMER["Contract"] in [0, 1, 2]

def test_tenure_is_positive():
    assert HIGH_RISK_CUSTOMER["tenure"] > 0
    assert LOW_RISK_CUSTOMER["tenure"] > 0

def test_charges_are_positive():
    assert HIGH_RISK_CUSTOMER["MonthlyCharges"] > 0
    assert LOW_RISK_CUSTOMER["TotalCharges"] > 0

def test_api_health():
    from fastapi.testclient import TestClient
    from src.serving.api import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

def test_api_predict_returns_valid_structure():
    from unittest.mock import patch, MagicMock
    import numpy as np

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    with patch("src.serving.api.model", mock_model):
        from fastapi.testclient import TestClient
        from src.serving.api import app
        client = TestClient(app)
        response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "risk_level" in data
        assert data["risk_level"] in ["Low", "Medium", "High"]
        assert 0 <= data["churn_probability"] <= 1

def test_api_predict_low_risk_structure():
    from unittest.mock import patch, MagicMock
    import numpy as np

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

    with patch("src.serving.api.model", mock_model):
        from fastapi.testclient import TestClient
        from src.serving.api import app
        client = TestClient(app)
        response = client.post("/predict", json=LOW_RISK_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] == "Low"