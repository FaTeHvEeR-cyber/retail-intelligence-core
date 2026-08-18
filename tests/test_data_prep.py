"""
Unit tests for data_prep module.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.data_prep import (
    ColumnMapping,
    DataPrepConfig,
    clean_and_impute,
    add_calendar_features,
    add_competition_and_promo_dynamics,
    add_lag_and_rolling_features,
    engineer_features,
    chronological_train_val_split,
)


@pytest.fixture
def sample_raw_data():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df1 = pd.DataFrame({
        "Store": [1] * 60,
        "Date": dates,
        "Sales": np.random.randint(1000, 5000, size=60),
        "Customers": np.random.randint(100, 500, size=60),
        "Open": [1] * 58 + [0, 1],
        "Promo": [0, 1] * 30,
        "DayOfWeek": [d.dayofweek + 1 for d in dates],
        "StateHoliday": ["0"] * 60,
        "SchoolHoliday": [0] * 60,
        "StoreType": ["a"] * 60,
        "Assortment": ["a"] * 60,
        "CompetitionDistance": [500.0] * 50 + [np.nan] * 10,
        "CompetitionOpenSinceMonth": [3.0] * 60,
        "CompetitionOpenSinceYear": [2020.0] * 60,
        "Promo2": [1] * 60,
        "Promo2SinceWeek": [10.0] * 60,
        "Promo2SinceYear": [2021.0] * 60,
        "PromoInterval": ["Jan,Apr,Jul,Oct"] * 60,
    })
    return df1


def test_clean_and_impute(sample_raw_data):
    cols = ColumnMapping()
    cleaned = clean_and_impute(sample_raw_data, cols)
    
    # Check that closed store row (Open == 0) was dropped
    assert (cleaned["Open"] == 0).sum() == 0
    # Check that missing competition distance was imputed
    assert cleaned["CompetitionDistance"].isna().sum() == 0
    # Check anomaly flag exists
    assert "is_zero_sales_anomaly" in cleaned.columns


def test_calendar_features(sample_raw_data):
    cols = ColumnMapping()
    cleaned = clean_and_impute(sample_raw_data, cols)
    df = add_calendar_features(cleaned, cols)
    
    assert "Year" in df.columns
    assert "Month" in df.columns
    assert "Day" in df.columns
    assert "Quarter" in df.columns
    assert "WeekOfYear" in df.columns
    assert "IsWeekend" in df.columns
    assert "IsPromo" in df.columns


def test_lag_and_rolling_features(sample_raw_data):
    cols = ColumnMapping()
    cfg = DataPrepConfig(lag_days=[7, 14], rolling_windows=[7])
    cleaned = clean_and_impute(sample_raw_data, cols)
    df = add_lag_and_rolling_features(cleaned, cols, cfg)
    
    assert "sales_lag_7" in df.columns
    assert "sales_lag_14" in df.columns
    assert "sales_rolling_mean_7" in df.columns
    assert "sales_rolling_std_7" in df.columns


def test_chronological_split(sample_raw_data):
    cols = ColumnMapping()
    cleaned = clean_and_impute(sample_raw_data, cols)
    train, val = chronological_train_val_split(cleaned, cols, validation_weeks=2)
    
    assert len(train) > 0
    assert len(val) > 0
    assert train["Date"].max() < val["Date"].min()
