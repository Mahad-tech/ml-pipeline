# End-to-End ML Pipeline

A modular, production-grade machine learning pipeline demonstrating MLOps best practices.
Includes pluggable use-case modules, experiment tracking, a REST API, and CI/CD.

## Architecture
## Use Cases

| Module | Dataset | Model | Metric |
|---|---|---|---|
| Churn Prediction | Telco Customer Churn (7k rows) | XGBoost + Logistic Regression | ROC-AUC: 0.86 |
| Sales Forecasting | Store Item Demand (913k rows) | Prophet | MAPE: 23% |
| Sentiment Analysis | IMDB Reviews | DistilBERT | coming soon |

## Stack
- **ML:** scikit-learn, XGBoost, Prophet
- **Experiment Tracking:** MLflow
- **Serving:** FastAPI + uvicorn
- **Testing:** pytest (14 tests)
- **CI/CD:** GitHub Actions

## Quickstart

```bash
git clone https://github.com/Mahad-tech/ml-pipeline.git
cd ml-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python main.py                                      # churn pipeline
python use_cases/forecasting/train_forecast.py      # forecasting pipeline
python use_cases/forecasting/forecast_api.py        # 7-day forecast preview

mlflow ui                                           # experiments at localhost:5000
uvicorn src.serving.api:app --reload               # API at localhost:8000
pytest tests/ -v                                    # run all 14 tests
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /predict | POST | Churn probability + risk level |
| /forecast?store=1&item=1&days=7 | GET | Sales forecast for N days |
| /health | GET | Health check |
| /docs | GET | Interactive Swagger UI |

## Key Design Decisions
- `src/` contains the pipeline engine; `use_cases/` contains swappable business modules
- Time series split by date only — no random splits to prevent data leakage
- `class_weight='balanced'` handles 27% churn class imbalance
- sklearn `Pipeline` ensures consistent scaling at train and predict time
- Prophet chosen over ARIMA for automatic multi-seasonality handling
