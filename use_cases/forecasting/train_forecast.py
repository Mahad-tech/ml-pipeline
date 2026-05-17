import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/forecasting/train.csv", parse_dates=["date"])
print(f"Loaded {df.shape[0]:,} rows | date range: {df['date'].min()} → {df['date'].max()}")
print(f"Stores: {df['store'].nunique()} | Items: {df['item'].nunique()}")

# ── Focus on one store + one item (store=1, item=1) ────────────────────────
series = (
    df[(df["store"] == 1) & (df["item"] == 1)]
    .groupby("date")["sales"]
    .sum()
    .reset_index()
    .rename(columns={"date": "ds", "sales": "y"})
)
print(f"\nWorking series shape: {series.shape}")
print(series.head())

# ── Train/test split (last 90 days = test) ─────────────────────────────────
# IMPORTANT: for time series we NEVER random split
# We always split by time — past for training, future for testing
cutoff = series["ds"].max() - pd.Timedelta(days=90)
train = series[series["ds"] <= cutoff]
test  = series[series["ds"] >  cutoff]
print(f"\nTrain: {train.shape[0]} days | Test: {test.shape[0]} days")
print(f"Train ends: {train['ds'].max()} | Test starts: {test['ds'].min()}")

# ── Train Prophet model ────────────────────────────────────────────────────
with mlflow.start_run(run_name="forecasting-prophet-store1-item1"):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(train)

    # Predict on test period
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Extract test predictions
    test_forecast = forecast[forecast["ds"].isin(test["ds"])][["ds", "yhat"]]
    merged = test.merge(test_forecast, on="ds")

    # Metrics
    mae  = mean_absolute_error(merged["y"], merged["yhat"])
    rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
    mape = (abs(merged["y"] - merged["yhat"]) / merged["y"]).mean() * 100

    mlflow.log_param("model", "Prophet")
    mlflow.log_param("store", 1)
    mlflow.log_param("item", 1)
    mlflow.log_param("train_days", len(train))
    mlflow.log_param("test_days", len(test))
    mlflow.log_param("changepoint_prior_scale", 0.05)
    mlflow.log_metric("mae", round(mae, 4))
    mlflow.log_metric("rmse", round(rmse, 4))
    mlflow.log_metric("mape", round(mape, 4))

    print(f"\n=== Prophet Results ===")
    print(f"MAE:  {mae:.2f}  (on average, off by {mae:.1f} units per day)")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}% (on average, {mape:.1f}% off from actual)")

    # Save forecast plot
    os.makedirs("models", exist_ok=True)
    fig = model.plot(forecast)
    plt.title("Prophet Forecast — Store 1, Item 1")
    plt.savefig("models/forecast_plot.png")
    plt.close()
    mlflow.log_artifact("models/forecast_plot.png")
    print("Forecast plot saved to models/forecast_plot.png")

    # Save components plot (trend + weekly + yearly)
    fig2 = model.plot_components(forecast)
    plt.savefig("models/forecast_components.png")
    plt.close()
    mlflow.log_artifact("models/forecast_components.png")
    print("Components plot saved to models/forecast_components.png")

print("\nForecasting pipeline complete.")
