import pandas as pd
from sklearn.preprocessing import LabelEncoder

def transform_churn(df: pd.DataFrame) -> tuple:
    df = df.copy()

    # Drop unhelpful ID column
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges — it's stored as string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode all object columns as numbers
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y
