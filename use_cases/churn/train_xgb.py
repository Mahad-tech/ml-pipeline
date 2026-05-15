import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from src.ingestion.loader import load_csv
from src.features.transformer import transform_churn
import pickle

df = load_csv("data/raw/churn/churn.csv")
X, y = transform_churn(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="churn-xgboost"):
    model = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 4)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model")
    print(f"XGBoost — Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

os.makedirs("models", exist_ok=True)
with open("models/churn_xgb.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to models/churn_xgb.pkl")
