from src.ingestion.loader import load_csv
from src.features.transformer import transform_churn
from src.training.trainer import train_churn_model
from src.evaluation.metrics import evaluate

DATA_PATH = "data/raw/churn/churn.csv"

if __name__ == "__main__":
    print("=== Step 1: Load Data ===")
    df = load_csv(DATA_PATH)

    print("\n=== Step 2: Feature Engineering ===")
    X, y = transform_churn(df)
    print(f"Features shape: {X.shape}")

    print("\n=== Step 3: Train Model ===")
    model, X_test, y_test = train_churn_model(X, y)

    print("\n=== Step 4: Evaluate ===")
    evaluate(model, X_test, y_test)

    print("\nPipeline complete. Run 'mlflow ui' to see experiment tracking.")
