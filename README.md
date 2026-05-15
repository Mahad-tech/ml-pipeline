# End-to-End ML Pipeline

A modular, production-grade machine learning pipeline demonstrating MLOps best practices.
Includes pluggable use-case modules, experiment tracking, a REST API, and CI/CD.

## Architecture
## Use Cases
| Module | Dataset | Model | ROC-AUC |
|---|---|---|---|
| Churn Prediction | Telco Customer Churn (7k rows) | XGBoost + Logistic Regression | 0.86 |
| Sales Forecasting | Store Item Demand | ARIMA + Prophet | coming soon |
| Sentiment Analysis | IMDB Reviews | DistilBERT | coming soon |

## Stack
- **ML:** scikit-learn, XGBoost
- **Experiment Tracking:** MLflow
- **Serving:** FastAPI + uvicorn
- **Testing:** pytest (7 tests)
- **CI/CD:** GitHub Actions

## Quickstart

```bash
git clone https://github.com/Mahad-tech/ml-pipeline.git
cd ml-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py          # run full pipeline
mlflow ui               # view experiments at localhost:5000
uvicorn src.serving.api:app --reload  # start API at localhost:8000
pytest tests/ -v        # run all tests
```

## API

`POST /predict` — returns churn probability and risk level (Low / Medium / High)

`GET /docs` — interactive Swagger UI

## Key Design Decisions
- `src/` contains the pipeline engine; `use_cases/` contains swappable business modules
- `class_weight='balanced'` on Logistic Regression to handle 27% churn class imbalance
- sklearn `Pipeline` ensures scaling is applied consistently at train and predict time
- Feature schema must match training order — documented in `src/serving/api.py`
