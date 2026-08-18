"""
Unit tests for src/train_models.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train_models import (
    calculate_rmspe,
    prepare_features,
    build_preprocessor,
    train_and_cross_validate,
    train_forecasting_models,
    EXCLUDE_COLUMNS,
)


def test_calculate_rmspe_standard():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    # errors: 10/100=0.1, -10/200=-0.05, 0/300=0 -> squares: 0.01, 0.0025, 0 -> mean: 0.00416666... -> sqrt: ~0.0645497
    rmspe, excluded = calculate_rmspe(y_true, y_pred)
    assert excluded == 0
    assert 0.064 < rmspe < 0.065


def test_calculate_rmspe_zero_handling():
    y_true = np.array([0.0, 100.0, 0.0, 200.0])
    y_pred = np.array([10.0, 110.0, 50.0, 200.0])
    rmspe, excluded = calculate_rmspe(y_true, y_pred)
    assert excluded == 2
    assert rmspe > 0.0


def test_prepare_features_leakage_exclusion():
    df = pd.DataFrame({
        "Store": [1, 2],
        "Date": pd.to_datetime(["2015-01-01", "2015-01-02"]),
        "Sales": [5000, 6000],
        "Customers": [500, 600],
        "Open": [1, 1],
        "is_zero_sales_anomaly": [0, 0],
        "Promo": [1, 0],
        "StoreType": ["a", "b"],
        "CompetitionDistance": [100.0, 200.0],
    })

    X, y, num_cols, cat_cols = prepare_features(df, exclude_anomalies=True)

    for col in EXCLUDE_COLUMNS:
        assert col not in X.columns
    assert "Customers" not in X.columns
    assert "Sales" not in X.columns
    assert "Open" not in X.columns

    assert "Promo" in num_cols or "Promo" in X.columns
    assert "StoreType" in cat_cols
    assert len(y) == 2


def test_build_preprocessor_fit_transform():
    num_cols = ["Promo", "CompetitionDistance"]
    cat_cols = ["StoreType"]

    df = pd.DataFrame({
        "Promo": [1, 0, np.nan],
        "CompetitionDistance": [100.0, np.nan, 300.0],
        "StoreType": ["a", "b", "a"],
    })

    prep = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    transformed = prep.fit_transform(df)

    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()


def test_train_forecasting_models_synthetic(tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()

    # Generate synthetic time-series data
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
    df.to_parquet(train_file)

    manifest = train_forecasting_models(
        data_dir=str(data_dir),
        models_dir=str(models_dir),
        config_path="configs/rossmann_mapping.yaml",
        n_splits=2,
        sample_frac=None,
        exclude_anomalies=True,
    )

    assert "models" in manifest
    assert "ridge" in manifest["models"]
    assert "xgboost" in manifest["models"]
    assert "mlp" in manifest["models"]

    for m_name, m_info in manifest["models"].items():
        assert Path(m_info["artifact_path"]).exists()
