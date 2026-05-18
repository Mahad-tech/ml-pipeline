# ml-pipeline

> Predicts which telecom customers are likely to cancel their subscription,
> so retention teams can intervene before they leave.

End-to-end ML pipeline with experiment tracking, a REST API, and CI/CD.
Trained on 7,043 real customer records — **ROC-AUC 0.86**.

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 75.7% | **0.86** |
| XGBoost | **80.6%** | 0.85 |

XGBoost wins on accuracy. Logistic Regression wins on ROC-AUC because
`class_weight='balanced'` prioritises catching churners over overall correctness —
the right trade-off when false negatives cost more than false positives.

---

## Architecture

```
CSV / API / DB
      │
      ▼
Data Ingestion  ──►  Feature Engineering  ──►  Model Training  ──►  Evaluation
(loader.py)          (transformer.py)           (trainer.py)         (metrics.py)
                                                      │
                                                      ▼
                                               Serving (FastAPI)
                                               POST /predict
```

## Quickstart

```bash
git clone https://github.com/Mahad-tech/ml-pipeline.git
cd ml-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python main.py                         # run full pipeline
mlflow ui                              # experiments → localhost:5000
uvicorn src.serving.api:app --reload   # API → localhost:8000
pytest tests/ -v                       # 14 tests
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `POST /predict` | POST | Churn probability + risk level (Low / Medium / High) |
| `GET /health` | GET | Health check |
| `GET /docs` | GET | Interactive Swagger UI |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 65.0, "Contract": "Month-to-month", ...}'
```

## Design Decisions

- `src/` is the pipeline engine — `use_cases/` holds swappable business modules
- `class_weight='balanced'` handles the 27% churn class imbalance
- sklearn `Pipeline` ensures scaling is applied at train and predict time consistently
- Feature schema is fixed to training order — all 19 features documented in `src/serving/api.py`

## Dataset

Telco Customer Churn — 7,043 customers, 19 features, 27% churn rate.
Source: IBM Sample Dataset via [Kaggle].

## Stack

`scikit-learn` · `XGBoost` · `MLflow` · `FastAPI` · `pytest` · `GitHub Actions`