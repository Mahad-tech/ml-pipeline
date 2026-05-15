import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def train_churn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="churn-logistic-regression-v2"):
        # Pipeline: scale first, then fit — this is production best practice
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_param("model", "LogisticRegression-v2")
        mlflow.log_param("scaled", True)
        mlflow.log_param("max_iter", 2000)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

    return pipeline, X_test, y_test
