import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# pyright: ignore [reportMissingImports]
from src.fraud_detection import (
    ReturnSimConfig,
    AutoencoderConfig,
    FEATURE_COLUMNS,
    generate_synthetic_returns,
    build_feature_matrix,
    chronological_train_monitor_split,
    train_autoencoder,
    compute_reconstruction_error,
    flag_anomalies,
    evaluate_against_ground_truth,
    run_pipeline,
)


@pytest.fixture
def small_sim_config():
    return ReturnSimConfig(
        n_transactions=500,
        anomaly_rate=0.06,
        n_stores=5,
        n_customers=50,
        lookback_days=30,
        random_seed=42,
    )


@pytest.fixture
def fast_ae_config():
    return AutoencoderConfig(
        hidden_layer_sizes=(8, 4, 8),
        max_iter=50,
        random_state=42,
        train_fraction=0.7,
    )


def test_generate_synthetic_returns(small_sim_config):
    df = generate_synthetic_returns(small_sim_config)
    assert len(df) == 500
    expected_cols = {
        "transaction_id",
        "store_id",
        "customer_id",
        "timestamp",
        "return_amount",
        "return_frequency_24h",
        "days_since_purchase",
        "receipt_verified",
        "customer_tenure_days",
        "is_synthetic_anomaly",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["is_synthetic_anomaly"].nunique() == 2
    n_anomalies = df["is_synthetic_anomaly"].sum()
    assert n_anomalies == 30  # 500 * 0.06 = 30
    assert df["timestamp"].is_monotonic_increasing


def test_build_feature_matrix(small_sim_config):
    df = generate_synthetic_returns(small_sim_config)
    feats = build_feature_matrix(df)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert "is_synthetic_anomaly" not in feats.columns
    assert "transaction_id" not in feats.columns
    np.testing.assert_allclose(feats["return_amount_log"], np.log1p(df["return_amount"]))


def test_chronological_train_monitor_split(small_sim_config):
    df = generate_synthetic_returns(small_sim_config)
    feats = build_feature_matrix(df)
    df_train, df_monitor, X_train, X_monitor = chronological_train_monitor_split(
        df, feats, train_fraction=0.7
    )
    assert len(df_train) == 350
    assert len(df_monitor) == 150
    assert len(X_train) == 350
    assert len(X_monitor) == 150
    assert df_train["timestamp"].max() <= df_monitor["timestamp"].min()


def test_train_autoencoder_and_reconstruction(small_sim_config, fast_ae_config):
    df = generate_synthetic_returns(small_sim_config)
    feats = build_feature_matrix(df)
    _, _, X_train, X_monitor = chronological_train_monitor_split(
        df, feats, train_fraction=fast_ae_config.train_fraction
    )

    model, scaler = train_autoencoder(X_train, fast_ae_config)
    assert model is not None
    assert scaler is not None

    train_errors = compute_reconstruction_error(model, scaler, X_train)
    monitor_errors = compute_reconstruction_error(model, scaler, X_monitor)

    assert len(train_errors) == len(X_train)
    assert len(monitor_errors) == len(X_monitor)
    assert (train_errors >= 0).all()
    assert (monitor_errors >= 0).all()


def test_flag_anomalies_threshold_isolation():
    train_errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    scored_errors = np.array([0.05, 0.5, 0.95, 1.5])

    # 90th percentile of train is 0.91
    flags, thresh = flag_anomalies(train_errors, scored_errors, percentile=90.0)
    assert thresh == pytest.approx(float(np.percentile(train_errors, 90.0)))
    # Only 0.95 and 1.5 should be flagged
    np.testing.assert_array_equal(flags, [False, False, True, True])


def test_evaluate_against_ground_truth():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    metrics = evaluate_against_ground_truth(y_true, y_pred)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["n_flagged"] == 3
    assert metrics["n_actual_anomalies"] == 3
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_run_pipeline_end_to_end(tmp_path, small_sim_config, fast_ae_config):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    summary = run_pipeline(
        sim_config=small_sim_config,
        ae_config=fast_ae_config,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    assert "n_train" in summary
    assert "n_monitor" in summary
    assert "threshold_p95" in summary
    assert "threshold_p99" in summary
    assert "eval_p95_diagnostic_only" in summary
    assert "eval_p99_diagnostic_only" in summary

    # Verify persisted model artifacts
    assert (models_dir / "fraud_autoencoder_latest.joblib").exists()
    assert (models_dir / "fraud_scaler_latest.joblib").exists()

    # Verify reports
    alerts_path = reports_dir / "fraud_alerts.json"
    metrics_path = reports_dir / "fraud_detection_metrics.json"
    assert alerts_path.exists()
    assert metrics_path.exists()

    with open(alerts_path) as f:
        alerts = json.load(f)
    assert len(alerts) == summary["n_monitor"]
    assert "reconstruction_error" in alerts[0]
    assert "severity" in alerts[0]
    assert "flagged_p95" in alerts[0]
    assert "flagged_p99" in alerts[0]
    assert "is_synthetic_anomaly" not in alerts[0]  # production-ready alert feed
