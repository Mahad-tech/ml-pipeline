import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import pandas as pd

DATA_PATH = "data/raw/forecasting/train.csv"

def test_data_loads():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    assert df.shape[0] > 100000
    assert "date" in df.columns
    assert "sales" in df.columns
    assert "store" in df.columns
    assert "item" in df.columns

def test_data_has_no_nulls():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0

def test_sales_are_positive():
    df = pd.read_csv(DATA_PATH)
    assert (df["sales"] >= 0).all()

def test_store_count():
    df = pd.read_csv(DATA_PATH)
    assert df["store"].nunique() == 10

def test_item_count():
    df = pd.read_csv(DATA_PATH)
    assert df["item"].nunique() == 50

def test_time_split_logic():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    series = (
        df[(df["store"] == 1) & (df["item"] == 1)]
        .groupby("date")["sales"].sum()
        .reset_index()
        .rename(columns={"date": "ds", "sales": "y"})
    )
    cutoff = series["ds"].max() - pd.Timedelta(days=90)
    train  = series[series["ds"] <= cutoff]
    test   = series[series["ds"] >  cutoff]
    assert len(test) == 90
    assert train["ds"].max() < test["ds"].min()  # no data leakage

def test_forecast_output_structure():
    from use_cases.forecasting.forecast_api import train_and_predict
    result = train_and_predict(
        store=1, item=1, days_ahead=7,
        df_path=DATA_PATH
    )
    assert "predictions" in result
    assert len(result["predictions"]) == 7
    assert "date" in result["predictions"][0]
    assert "predicted_sales" in result["predictions"][0]
    assert result["predictions"][0]["predicted_sales"] >= 0
