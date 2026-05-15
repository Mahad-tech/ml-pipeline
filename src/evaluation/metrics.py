from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print("=== Confusion Matrix ===")
    cm = pd.DataFrame(
        confusion_matrix(y_test, preds),
        index=["Actual No", "Actual Yes"],
        columns=["Pred No", "Pred Yes"]
    )
    print(cm)
