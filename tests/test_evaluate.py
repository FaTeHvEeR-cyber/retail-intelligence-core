"""
Unit tests for src/evaluate.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_model_pipeline, run_evaluation
from src.train_models import train_forecasting_models


def test_evaluate_model_pipeline():
    X_val = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0]})
    y_val = pd.Series([100.0, 200.0, 300.0, 400.0])

    dummy = Pipeline([("model", DummyRegressor(strategy="mean"))])
    dummy.fit(X_val, y_val)

    result = evaluate_model_pipeline("dummy", dummy, X_val, y_val, train_time_sec=1.5)

    assert result["model_name"] == "dummy"
    assert result["rmspe"] > 0.0
    assert result["mae"] > 0.0
    assert result["train_time_sec"] == 1.5
    assert result["inference_time_sec"] >= 0.0
    assert result["rows_excluded_zero_sales"] == 0


def test_run_evaluation_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()
    reports_dir.mkdir()

    dates = pd.date_range("2015-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "Store": np.tile([1, 2], 30),
        "Date": np.repeat(dates[:30], 2),
        "Sales": np.random.randint(2000, 10000, size=60),
        "Customers": np.random.randint(200, 800, size=60),
        "Open": np.ones(60, dtype=int),
        "Promo": np.random.randint(0, 2, size=60),
        "StoreType": np.random.choice(["a", "b", "c"], size=60),
        "Assortment": np.random.choice(["a", "b"], size=60),
        "CompetitionDistance": np.random.uniform(100, 5000, size=60),
        "is_zero_sales_anomaly": np.zeros(60, dtype=int),
        "sales_lag_7": np.random.uniform(2000, 10000, size=60),
    })

    train_file = data_dir / "train_processed.parquet"
    val_file = data_dir / "val_processed.parquet"
    df.iloc[:40].to_parquet(train_file)
    df.iloc[40:].to_parquet(val_file)

    train_forecasting_models(
        data_dir=str(data_dir),
        models_dir=str(models_dir),
        config_path="configs/rossmann_mapping.yaml",
        n_splits=2,
    )

    out_json = reports_dir / "comparison_metrics.json"
    report = run_evaluation(
        data_dir=str(data_dir),
        models_dir=str(models_dir),
        out_path=str(out_json),
    )

    assert out_json.exists()
    assert report["validation_rows"] == 20
    assert len(report["models"]) == 3
    assert report["models"][0]["rmspe"] <= report["models"][1]["rmspe"]
