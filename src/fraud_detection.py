"""
src/fraud_detection.py
======================
Module 5: Point-of-Sale (POS) Anomaly & Fraud Simulation.

Responsibilities:
    - Simulate a realistic stream of POS return transactions, injecting a
      minority of fraud-like/abusive patterns (return abuse, unverified
      receipts, wardrobing, brand-new-account abuse).
    - Build a leak-free feature matrix from the transaction stream.
    - Train an unsupervised Autoencoder-style ANN to learn "normal"
      transaction structure, then score every transaction by reconstruction
      error (MSE between input and reconstruction).
    - Flag the top 95th / 99th percentile reconstruction-error transactions
      as high-risk anomalies.
    - Serialize the trained model + scaler, and persist a flagged-alert
      report to reports/.

Design principles (consistent with Module 1's leak-free philosophy):
    - The anomaly-score threshold is learned ONLY from the training window's
      reconstruction-error distribution, then applied forward to new
      (monitoring-window) transactions -- never computed on the same data
      it is judging.
    - The synthetic `is_synthetic_anomaly` ground-truth label exists ONLY
      for offline evaluation/QA of the detector. It is never included in
      the feature matrix used to train or score the model -- in production
      this label would not exist at POS time.

Model architecture note:
    This module uses a symmetric-bottleneck `MLPRegressor`
    (input -> hidden -> bottleneck -> hidden -> output, trained to
    reconstruct its own input). This is architecturally equivalent to an
    encoder/decoder with a compressive bottleneck and MSE reconstruction loss,
    and provides lightweight, reliable CPU training.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class ReturnSimConfig:
    """Controls the synthetic POS return-transaction generator."""

    n_transactions: int = 20_000
    anomaly_rate: float = 0.03          # ~3% of transactions are fraud-like
    n_stores: int = 50
    n_customers: int = 5_000
    lookback_days: int = 180            # transactions span the last N days
    random_seed: int = 42


@dataclass
class AutoencoderConfig:
    """Controls the reconstruction-error anomaly detector."""

    hidden_layer_sizes: tuple[int, ...] = (16, 8, 16)  # encoder -> bottleneck -> decoder
    activation: str = "relu"
    solver: str = "adam"
    learning_rate_init: float = 1e-3
    max_iter: int = 400
    random_state: int = 42
    threshold_percentiles: tuple[float, float] = (95.0, 99.0)
    train_fraction: float = 0.7         # chronological: first 70% trains the detector


FEATURE_COLUMNS: list[str] = [
    "return_amount_log",
    "return_frequency_24h",
    "days_since_purchase",
    "receipt_verified",
    "customer_tenure_days",
]


# ----------------------------------------------------------------------------
# Synthetic Data Generation
# ----------------------------------------------------------------------------
def generate_synthetic_returns(config: ReturnSimConfig) -> pd.DataFrame:
    """
    Simulate a POS return-transaction stream with an injected minority of
    fraud-like patterns.

    Normal transactions:
        - Modest return amounts, low same-day return frequency, purchases
          returned within a normal 1-30 day window, receipts usually
          verified, tenured customer accounts.

    Injected anomaly patterns (roughly evenly split across anomalies):
        A) Return-abuse: high return_frequency_24h (serial returner).
        B) Receipt fraud: unverified receipt + high return amount.
        C) Wardrobing: very short days_since_purchase + high amount on a
           brand-new account (low customer_tenure_days).

    Returns:
        DataFrame with one row per transaction, including the ground-truth
        `is_synthetic_anomaly` label (evaluation-only -- never a feature).
    """
    rng = np.random.default_rng(config.random_seed)
    n_anomaly = int(round(config.n_transactions * config.anomaly_rate))
    n_normal = config.n_transactions - n_anomaly

    def _random_timestamps(n: int) -> np.ndarray:
        offsets = rng.integers(0, config.lookback_days * 24 * 60, size=n)
        base = datetime.now() - timedelta(days=config.lookback_days)
        return np.array([base + timedelta(minutes=int(m)) for m in offsets])

    # --- Normal transactions ---
    normal = pd.DataFrame(
        {
            "store_id": rng.integers(1, config.n_stores + 1, size=n_normal),
            "customer_id": rng.integers(1, config.n_customers + 1, size=n_normal),
            "timestamp": _random_timestamps(n_normal),
            "return_amount": rng.lognormal(mean=3.0, sigma=0.6, size=n_normal).clip(2, 400),
            "return_frequency_24h": rng.poisson(lam=0.3, size=n_normal).clip(0, 2),
            "days_since_purchase": rng.integers(1, 31, size=n_normal),
            "receipt_verified": rng.choice([1, 0], size=n_normal, p=[0.94, 0.06]),
            "customer_tenure_days": rng.integers(60, 3000, size=n_normal),
            "is_synthetic_anomaly": 0,
        }
    )

    # --- Anomalous transactions: split across three patterns ---
    n_a = n_anomaly // 3
    n_b = n_anomaly // 3
    n_c = n_anomaly - n_a - n_b

    pattern_a = pd.DataFrame(  # serial returner / return abuse
        {
            "store_id": rng.integers(1, config.n_stores + 1, size=n_a),
            "customer_id": rng.integers(1, config.n_customers + 1, size=n_a),
            "timestamp": _random_timestamps(n_a),
            "return_amount": rng.lognormal(mean=3.2, sigma=0.7, size=n_a).clip(2, 500),
            "return_frequency_24h": rng.integers(3, 9, size=n_a),
            "days_since_purchase": rng.integers(1, 31, size=n_a),
            "receipt_verified": rng.choice([1, 0], size=n_a, p=[0.6, 0.4]),
            "customer_tenure_days": rng.integers(30, 2500, size=n_a),
            "is_synthetic_anomaly": 1,
        }
    )

    pattern_b = pd.DataFrame(  # receipt / high-value fraud
        {
            "store_id": rng.integers(1, config.n_stores + 1, size=n_b),
            "customer_id": rng.integers(1, config.n_customers + 1, size=n_b),
            "timestamp": _random_timestamps(n_b),
            "return_amount": rng.lognormal(mean=5.3, sigma=0.5, size=n_b).clip(150, 2000),
            "return_frequency_24h": rng.poisson(lam=0.5, size=n_b).clip(0, 3),
            "days_since_purchase": rng.integers(1, 60, size=n_b),
            "receipt_verified": rng.choice([1, 0], size=n_b, p=[0.15, 0.85]),
            "customer_tenure_days": rng.integers(30, 3000, size=n_b),
            "is_synthetic_anomaly": 1,
        }
    )

    pattern_c = pd.DataFrame(  # wardrobing on brand-new accounts
        {
            "store_id": rng.integers(1, config.n_stores + 1, size=n_c),
            "customer_id": rng.integers(1, config.n_customers + 1, size=n_c),
            "timestamp": _random_timestamps(n_c),
            "return_amount": rng.lognormal(mean=4.8, sigma=0.5, size=n_c).clip(80, 1200),
            "return_frequency_24h": rng.poisson(lam=0.4, size=n_c).clip(0, 3),
            "days_since_purchase": rng.integers(0, 3, size=n_c),
            "receipt_verified": rng.choice([1, 0], size=n_c, p=[0.5, 0.5]),
            "customer_tenure_days": rng.integers(0, 7, size=n_c),
            "is_synthetic_anomaly": 1,
        }
    )

    df = pd.concat([normal, pattern_a, pattern_b, pattern_c], ignore_index=True)
    df = df.sample(frac=1.0, random_state=config.random_seed).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN-{i:07d}" for i in range(len(df))])

    logger.info(
        "Generated %d synthetic transactions (%d normal, %d injected anomalies -> %.2f%%)",
        len(df), n_normal, n_anomaly, 100 * n_anomaly / len(df),
    )
    return df


# ----------------------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the numeric feature matrix used by the anomaly detector.

    - return_amount is log1p-transformed: raw dollar returns are heavily
      right-skewed, and unscaled skew dominates MSE-based reconstruction
      loss, drowning out the other (informative) behavioral features.
    - receipt_verified is already binary (0/1).
    - `is_synthetic_anomaly`, if present, is intentionally excluded here --
      it is evaluation-only ground truth, never a model input.
    """
    feats = pd.DataFrame(index=df.index)
    feats["return_amount_log"] = np.log1p(df["return_amount"])
    feats["return_frequency_24h"] = df["return_frequency_24h"].astype(float)
    feats["days_since_purchase"] = df["days_since_purchase"].astype(float)
    feats["receipt_verified"] = df["receipt_verified"].astype(float)
    feats["customer_tenure_days"] = df["customer_tenure_days"].astype(float)
    return feats[FEATURE_COLUMNS]


def chronological_train_monitor_split(
    df: pd.DataFrame, feats: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by timestamp (not randomly): the detector is fit on the earlier
    window of "normal-dominated" traffic and then used to monitor/score the
    later window, mirroring how it would run in production.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    feats = feats.loc[df.index].reset_index(drop=True)
    cutoff_idx = int(len(df) * train_fraction)

    df_train, df_monitor = df.iloc[:cutoff_idx].copy(), df.iloc[cutoff_idx:].copy()
    X_train, X_monitor = feats.iloc[:cutoff_idx].copy(), feats.iloc[cutoff_idx:].copy()

    logger.info(
        "Chronological split -> train: %d txns (%s to %s), monitor: %d txns (%s to %s)",
        len(df_train), df_train["timestamp"].min(), df_train["timestamp"].max(),
        len(df_monitor), df_monitor["timestamp"].min(), df_monitor["timestamp"].max(),
    )
    return df_train, df_monitor, X_train, X_monitor


# ----------------------------------------------------------------------------
# Autoencoder: build, train, score
# ----------------------------------------------------------------------------
def _build_autoencoder(config: AutoencoderConfig) -> MLPRegressor:
    """
    Symmetric-bottleneck MLP trained to reconstruct its own (scaled) input.
    hidden_layer_sizes=(16, 8, 16) => 5-feature input -> 16 -> 8 (bottleneck)
    -> 16 -> 5-feature reconstruction.
    """
    return MLPRegressor(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        solver=config.solver,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        random_state=config.random_state,
        early_stopping=True,
        n_iter_no_change=15,
    )


def train_autoencoder(
    X_train: pd.DataFrame, config: AutoencoderConfig
) -> tuple[MLPRegressor, StandardScaler]:
    """
    Fit the scaler on the training window only, then train the autoencoder
    to reconstruct its own scaled input (X -> X).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)

    model = _build_autoencoder(config)
    logger.info(
        "Training autoencoder on %d transactions, architecture=%s",
        len(X_train), config.hidden_layer_sizes,
    )
    model.fit(X_train_scaled, X_train_scaled)
    return model, scaler


def compute_reconstruction_error(
    model: MLPRegressor, scaler: StandardScaler, X: pd.DataFrame
) -> np.ndarray:
    """Per-transaction reconstruction MSE (the anomaly score)."""
    X_scaled = scaler.transform(X.values)
    X_hat = model.predict(X_scaled)
    return np.mean((X_scaled - X_hat) ** 2, axis=1)


def flag_anomalies(
    train_errors: np.ndarray, scored_errors: np.ndarray, percentile: float
) -> tuple[np.ndarray, float]:
    """
    Threshold learned ONLY from `train_errors` (no leakage from the window
    being judged), applied to `scored_errors`.

    Returns:
        (boolean flag array, threshold value)
    """
    threshold = float(np.percentile(train_errors, percentile))
    flags = scored_errors > threshold
    return flags, threshold


# ----------------------------------------------------------------------------
# Evaluation (diagnostic only -- synthetic ground truth would not exist
# in a real deployment; this is here purely for offline QA of the detector).
# ----------------------------------------------------------------------------
def evaluate_against_ground_truth(y_true: np.ndarray, y_pred_flag: np.ndarray) -> dict:
    return {
        "precision": round(float(precision_score(y_true, y_pred_flag, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred_flag, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred_flag, zero_division=0)), 4),
        "n_flagged": int(y_pred_flag.sum()),
        "n_actual_anomalies": int(y_true.sum()),
    }


# ----------------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------------
def save_artifacts(model: MLPRegressor, scaler: StandardScaler, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, models_dir / f"fraud_autoencoder_{stamp}.joblib")
    joblib.dump(scaler, models_dir / f"fraud_scaler_{stamp}.joblib")
    # also save "latest" pointers for app.py to load without knowing the stamp
    joblib.dump(model, models_dir / "fraud_autoencoder_latest.joblib")
    joblib.dump(scaler, models_dir / "fraud_scaler_latest.joblib")
    logger.info("Saved autoencoder + scaler artifacts -> %s (stamp=%s)", models_dir, stamp)


def save_alert_report(
    df_monitor: pd.DataFrame,
    errors: np.ndarray,
    flags_95: np.ndarray,
    flags_99: np.ndarray,
    reports_dir: Path,
) -> Path:
    """Persist a POS Anomaly Alert Feed-ready report for app.py's dashboard tab."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_df = df_monitor.copy()
    report_df["reconstruction_error"] = errors
    report_df["flagged_p95"] = flags_95
    report_df["flagged_p99"] = flags_99
    report_df["severity"] = np.select(
        [flags_99, flags_95], ["high", "medium"], default="low"
    )
    report_df = report_df.sort_values("reconstruction_error", ascending=False)

    out_path = reports_dir / "fraud_alerts.json"
    report_df.drop(columns=["is_synthetic_anomaly"], errors="ignore").to_json(
        out_path, orient="records", date_format="iso", indent=2
    )
    logger.info("Saved POS anomaly alert feed -> %s (%d flagged at p95+)", out_path, int(flags_95.sum()))
    return out_path


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_pipeline(
    sim_config: ReturnSimConfig | None = None,
    ae_config: AutoencoderConfig | None = None,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
) -> dict:
    """Execute the full Module 5 pipeline end-to-end."""
    sim_config = sim_config or ReturnSimConfig()
    ae_config = ae_config or AutoencoderConfig()
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)

    df = generate_synthetic_returns(sim_config)
    feats = build_feature_matrix(df)
    df_train, df_monitor, X_train, X_monitor = chronological_train_monitor_split(
        df, feats, ae_config.train_fraction
    )

    model, scaler = train_autoencoder(X_train, ae_config)

    train_errors = compute_reconstruction_error(model, scaler, X_train)
    monitor_errors = compute_reconstruction_error(model, scaler, X_monitor)

    p95, p99 = ae_config.threshold_percentiles
    flags_95, thresh_95 = flag_anomalies(train_errors, monitor_errors, p95)
    flags_99, thresh_99 = flag_anomalies(train_errors, monitor_errors, p99)

    report_path = save_alert_report(df_monitor, monitor_errors, flags_95, flags_99, reports_dir)
    save_artifacts(model, scaler, models_dir)

    metrics_p95 = evaluate_against_ground_truth(df_monitor["is_synthetic_anomaly"].values, flags_95)
    metrics_p99 = evaluate_against_ground_truth(df_monitor["is_synthetic_anomaly"].values, flags_99)

    summary = {
        "n_train": len(df_train),
        "n_monitor": len(df_monitor),
        "threshold_p95": round(thresh_95, 5),
        "threshold_p99": round(thresh_99, 5),
        "eval_p95_diagnostic_only": metrics_p95,
        "eval_p99_diagnostic_only": metrics_p99,
        "alert_report_path": str(report_path),
    }
    logger.info("Module 5 pipeline summary: %s", json.dumps(summary, indent=2))

    metrics_path = reports_dir / "fraud_detection_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved detector metrics summary -> %s", metrics_path)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 5: POS Anomaly & Fraud Simulation")
    parser.add_argument("--n-transactions", type=int, default=20_000)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sim_cfg = ReturnSimConfig(
        n_transactions=args.n_transactions,
        anomaly_rate=args.anomaly_rate,
        random_seed=args.seed,
    )
    run_pipeline(sim_cfg, AutoencoderConfig(), args.models_dir, args.reports_dir)
