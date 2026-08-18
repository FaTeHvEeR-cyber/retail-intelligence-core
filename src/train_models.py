"""
train_models.py
===============
Module 3: Multi-Model Demand Forecasting & Benchmarking Engine.

Responsibilities:
    - Ingest processed training data (data/processed/train_processed.parquet).
    - Select feature set, strictly excluding target leakage (Customers, Open, is_zero_sales_anomaly, Date).
    - Build and train three architecturally distinct models:
        1. Ridge Regression (Linear Regularized Baseline)
        2. XGBoost Regressor (Gradient-Boosted Tree Ensemble)
        3. MLP Regressor (Deep Neural Non-Linear Architecture)
    - Perform TimeSeriesSplit (rolling temporal folds) cross-validation on train data.
    - Extract and record feature importances / linear coefficients.
    - Serialize fitted pipelines (model + encoders + imputers/scalers) and metadata into models/.

Design principles:
    - Zero data leakage: Chronological TimeSeriesSplit used exclusively.
    - Dataset-agnostic column mapping support.
    - Preprocessing (imputation, scaling, one-hot encoding) encapsulated in Scikit-Learn Pipelines.
    - Versioned artifact persistence using timestamps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data_prep import ColumnMapping, load_column_mapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Explicit feature selection & leakage exclusion policy
EXCLUDE_COLUMNS = {
    "Sales",  # Target variable
    "Customers",  # Target leakage: customer count is unknown at future forecast time
    "Open",  # Constant/closure indicator
    "is_zero_sales_anomaly",  # Label / anomaly flag
    "Date",  # Raw timestamp (temporal features extracted instead)
    "PromoInterval",  # Unused raw string
}


def calculate_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, int]:
    """
    Calculate Root Mean Square Percentage Error (RMSPE).
    Rows where y_true == 0 are excluded to prevent division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Clip negative predictions to 0 as retail sales cannot be negative
    y_pred = np.clip(y_pred, 0, None)

    valid_mask = y_true > 0
    excluded_count = int(np.sum(~valid_mask))

    if not np.any(valid_mask):
        return 0.0, excluded_count

    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]
    rmspe = float(np.sqrt(np.mean(((y_t - y_p) / y_t) ** 2)))
    return rmspe, excluded_count


def prepare_features(
    df: pd.DataFrame,
    mapping: ColumnMapping | None = None,
    exclude_anomalies: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Filter training records and partition columns into numeric and categorical features.

    Returns:
        X (DataFrame), y (Series), numeric_cols (list), categorical_cols (list)
    """
    sales_col = mapping.sales if mapping else "Sales"
    anomaly_col = "is_zero_sales_anomaly"

    data = df.copy()

    # Filter anomaly rows if requested
    if exclude_anomalies and anomaly_col in data.columns:
        n_before = len(data)
        data = data[data[anomaly_col] == 0].copy()
        logger.info("Excluded %d anomaly rows from training data.", n_before - len(data))

    # Ensure chronological order by Date / Store
    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)

    y = data[sales_col]

    feature_cols = [c for c in data.columns if c not in EXCLUDE_COLUMNS and c != sales_col]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c].dtype)]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    logger.info(
        "Prepared feature matrix with %d rows and %d features (%d numeric, %d categorical).",
        len(data),
        len(feature_cols),
        len(numeric_cols),
        len(categorical_cols),
    )
    logger.info("Categorical features: %s", categorical_cols)
    logger.info("Numeric features (%d total): %s", len(numeric_cols), numeric_cols)

    return data[feature_cols], y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build a leak-free ColumnTransformer for numeric and categorical features.
    """
    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipeline = Pipeline(num_steps)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers)


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """Retrieve post-transformation feature names from the fitted preprocessor."""
    feature_names = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_feature_names)
    return feature_names


def train_and_cross_validate(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train model pipeline with TimeSeriesSplit rolling temporal cross-validation,
    then fit on the entire training dataset.
    """
    logger.info("--- Training & Cross-Validating Model: %s ---", name)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_rmspe_scores: List[float] = []
    fold_mae_scores: List[float] = []

    fold = 1
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)

        rmspe, _ = calculate_rmspe(y_val.values, preds)
        mae = float(mean_absolute_error(y_val.values, np.clip(preds, 0, None)))

        fold_rmspe_scores.append(rmspe)
        fold_mae_scores.append(mae)
        logger.info(
            "[%s] Fold %d/%d - RMSPE: %.4f, MAE: %.2f (train_size=%d, val_size=%d)",
            name,
            fold,
            n_splits,
            rmspe,
            mae,
            len(train_idx),
            len(val_idx),
        )
        fold += 1

    # Full fit on entire training data
    logger.info("[%s] Fitting final model on complete training set (%d rows)...", name, len(X))
    start_time = time.time()
    pipeline.fit(X, y)
    train_time_sec = float(time.time() - start_time)

    meta = {
        "model_name": name,
        "n_train_rows": len(X),
        "cv_splits": n_splits,
        "cv_rmspe_folds": fold_rmspe_scores,
        "cv_rmspe_mean": float(np.mean(fold_rmspe_scores)),
        "cv_mae_mean": float(np.mean(fold_mae_scores)),
        "train_time_sec": round(train_time_sec, 2),
    }
    logger.info(
        "[%s] Final fit completed in %.2fs. Mean CV RMSPE: %.4f",
        name,
        train_time_sec,
        meta["cv_rmspe_mean"],
    )
    return pipeline, meta


def extract_feature_importances(
    name: str,
    pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 25,
) -> Dict[str, float]:
    """Extract top feature importances or coefficients for interpretability."""
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            feat_imp = {feat: float(c) for feat, c in zip(feature_names, coefs)}
        else:
            return {}

        sorted_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)[:top_n])
        return sorted_imp
    except Exception as e:
        logger.warning("[%s] Could not extract feature importances: %s", name, str(e))
        return {}


def train_forecasting_models(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    config_path: str = "configs/rossmann_mapping.yaml",
    n_splits: int = 3,
    sample_frac: float | None = None,
    exclude_anomalies: bool = True,
) -> Dict[str, Any]:
    """
    Main training execution function.
    Trains Ridge, XGBoost, and MLP Regressor models and persists artifacts.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    train_file = Path(data_dir) / "train_processed.parquet"
    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_file}")

    logger.info("Loading processed training data from %s", train_file)
    train_df = pd.read_parquet(train_file)

    mapping = load_column_mapping(config_path) if Path(config_path).exists() else None

    if sample_frac and 0.0 < sample_frac < 1.0:
        logger.info("Subsampling training data to %.1f%% for rapid training...", sample_frac * 100)
        train_df = train_df.sample(frac=sample_frac, random_state=42).sort_values("Date").reset_index(drop=True)

    X_train, y_train, numeric_cols, categorical_cols = prepare_features(
        train_df, mapping=mapping, exclude_anomalies=exclude_anomalies
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # 1. Model 1: Ridge Regression
    preprocessor_scaled = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor_scaled),
        ("model", Ridge(alpha=100.0, random_state=42)),
    ])
    ridge_fitted, ridge_meta = train_and_cross_validate("ridge", ridge_pipeline, X_train, y_train, n_splits=n_splits)

    # 2. Model 2: XGBoost Regressor
    preprocessor_unscaled = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor_unscaled),
        (
            "model",
            XGBRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ])
    xgb_fitted, xgb_meta = train_and_cross_validate("xgboost", xgb_pipeline, X_train, y_train, n_splits=n_splits)

    # 3. Model 3: MLP Regressor
    mlp_pipeline = Pipeline([
        ("preprocessor", preprocessor_scaled),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=30,
                batch_size=256,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
        ),
    ])
    mlp_fitted, mlp_meta = train_and_cross_validate("mlp", mlp_pipeline, X_train, y_train, n_splits=n_splits)

    # Extract feature names & importances
    fitted_prep = ridge_fitted.named_steps["preprocessor"]
    transformed_feature_names = get_feature_names(fitted_prep, numeric_cols, categorical_cols)

    ridge_imp = extract_feature_importances("ridge", ridge_fitted, transformed_feature_names)
    xgb_imp = extract_feature_importances("xgboost", xgb_fitted, transformed_feature_names)

    ridge_meta["top_feature_importance"] = ridge_imp
    xgb_meta["top_feature_importance"] = xgb_imp

    # Save artifacts
    artifacts = {
        "ridge": (ridge_fitted, ridge_meta, f"ridge_v{timestamp}.joblib"),
        "xgboost": (xgb_fitted, xgb_meta, f"xgboost_v{timestamp}.joblib"),
        "mlp": (mlp_fitted, mlp_meta, f"mlp_v{timestamp}.joblib"),
    }

    manifest = {
        "timestamp": timestamp,
        "raw_numeric_features": numeric_cols,
        "raw_categorical_features": categorical_cols,
        "transformed_feature_names": transformed_feature_names,
        "models": {},
    }

    for model_name, (model_obj, meta_obj, filename) in artifacts.items():
        file_path = models_path / filename
        joblib.dump(model_obj, file_path, compress=3)
        logger.info("Saved %s pipeline -> %s", model_name, file_path)

        meta_obj["artifact_path"] = str(file_path)
        meta_obj["filename"] = filename
        manifest["models"][model_name] = meta_obj

    manifest_path = models_path / "latest_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved training manifest -> %s", manifest_path)

    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 3: Demand Forecasting Model Training")
    parser.add_argument("--data", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save serialized models")
    parser.add_argument("--config", type=str, default="configs/rossmann_mapping.yaml", help="Path to column mapping config")
    parser.add_argument("--cv-splits", type=int, default=3, help="Number of TimeSeriesSplit folds")
    parser.add_argument("--sample-frac", type=float, default=None, help="Optional subsampling fraction for faster training (e.g. 0.2)")
    parser.add_argument("--include-anomalies", action="store_true", help="Include zero-sales anomalies in training data")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_forecasting_models(
        data_dir=args.data,
        models_dir=args.models_dir,
        config_path=args.config,
        n_splits=args.cv_splits,
        sample_frac=args.sample_frac,
        exclude_anomalies=not args.include_anomalies,
    )
