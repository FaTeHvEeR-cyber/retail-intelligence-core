import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# pyrefly: ignore [missing-import]
from app import (
    load_json,
    load_table,
    load_store_metadata,
    load_joblib,
    build_val_predictions,
    REPORTS_DIR,
    MODELS_DIR,
)


def test_load_json():
    # Existing file
    data = load_json("comparison_metrics.json")
    assert data is not None
    assert "models" in data
    assert len(data["models"]) >= 1

    # Non-existing file
    assert load_json("non_existent_file.json") is None


def test_load_table():
    # Load processed data
    val_df = load_table("val_processed")
    assert val_df is not None
    assert "Store" in val_df.columns
    assert "Sales" in val_df.columns
    assert len(val_df) > 0

    # Load from reports
    clusters_df = load_table("store_clusters", directory="reports")
    assert clusters_df is not None
    assert "cluster" in clusters_df.columns

    # Non-existent
    assert load_table("non_existent_table") is None


def test_load_store_metadata():
    store_meta = load_store_metadata()
    assert store_meta is not None
    assert "Store" in store_meta.columns
    assert "StoreType" in store_meta.columns
    assert "Assortment" in store_meta.columns


def test_load_joblib():
    # Existing joblib
    model = load_joblib("fraud_autoencoder_latest.joblib")
    assert model is not None

    # Non-existing
    assert load_joblib("non_existent_model.joblib") is None


def test_build_val_predictions():
    preds_df = build_val_predictions()
    assert preds_df is not None
    assert "Store" in preds_df.columns
    assert "Date" in preds_df.columns
    assert "Actual" in preds_df.columns
    pred_cols = [c for c in preds_df.columns if c.startswith("Predicted_")]
    assert len(pred_cols) >= 1
    # Check that predictions are valid non-negative numbers
    for c in pred_cols:
        assert (preds_df[c] >= 0).all()
        assert not preds_df[c].isna().any()


def test_reports_schema_integrity():
    # 1. hypothesis_tests.json
    hypo = load_json("hypothesis_tests.json")
    assert hypo is not None
    assert "tests" in hypo
    assert len(hypo["tests"]) == 3

    # 2. store_clusters.json
    cluster_meta = load_json("store_clusters.json")
    assert cluster_meta is not None
    assert cluster_meta["k"] == 4
    assert "cluster_profile_means" in cluster_meta

    # 3. fraud_alerts.json
    alerts = load_json("fraud_alerts.json")
    assert alerts is not None
    assert len(alerts) > 0
    assert "reconstruction_error" in alerts[0]
    assert "severity" in alerts[0]

    # 4. fraud_detection_metrics.json
    metrics = load_json("fraud_detection_metrics.json")
    assert metrics is not None
    assert "threshold_p95" in metrics
    assert "threshold_p99" in metrics
