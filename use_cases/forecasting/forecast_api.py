import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

def train_and_predict(store: int, item: int, days_ahead: int, df_path: str) -> dict:
    df = pd.read_csv(df_path, parse_dates=["date"])

    series = (
        df[(df["store"] == store) & (df["item"] == item)]
        .groupby("date")["sales"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "sales": "y"})
    )

    if series.empty:
        return {"error": f"No data found for store={store}, item={item}"}

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(series)

    future   = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    future_only = forecast.tail(days_ahead)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return {
        "store": store,
        "item": item,
        "days_ahead": days_ahead,
        "predictions": [
            {
                "date": str(row["ds"].date()),
                "predicted_sales": max(0, round(row["yhat"], 1)),
                "lower_bound": max(0, round(row["yhat_lower"], 1)),
                "upper_bound": max(0, round(row["yhat_upper"], 1))
            }
            for _, row in future_only.iterrows()
        ]
    }


if __name__ == "__main__":
    result = train_and_predict(
        store=1, item=1, days_ahead=7,
        df_path="data/raw/forecasting/train.csv"
    )
    print(f"\nStore {result['store']} | Item {result['item']} — Next 7 days forecast:")
    for p in result["predictions"]:
        print(f"  {p['date']}: {p['predicted_sales']} units "
              f"(range: {p['lower_bound']}–{p['upper_bound']})")
