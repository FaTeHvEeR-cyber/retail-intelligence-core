"""
evaluate.py
===========
Module 3b: Model Evaluation & Metric Benchmarking Suite.

Responsibilities:
    - Load chronological validation holdout dataset (data/processed/val_processed.parquet).
    - Load serialized model pipelines from models/.
    - Compute official competition metrics:
        - RMSPE (Primary Metric): Root Mean Square Percentage Error (excluding y_true == 0).
        - MAE: Mean Absolute Error.
        - RMSE: Root Mean Square Error.
        - R²: Coefficient of Determination.
    - Measure wall-clock inference latency per model.
    - Export consolidated benchmark report to reports/comparison_metrics.json.
    - Render a formatted ranked summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_prep import ColumnMapping, load_column_mapping
from src.train_models import EXCLUDE_COLUMNS, calculate_rmspe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_pipeline(
    name: str,
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate predictions and evaluate performance across all metrics.
    """
    logger.info("Evaluating model: %s on %d validation samples...", name, len(X_val))

    start_time = time.time()
    preds = pipeline.predict(X_val)
    inference_time_sec = float(time.time() - start_time)

    # Post-process retail predictions (cannot have negative sales)
    preds = np.clip(preds, 0, None)
    y_true = y_val.values

    rmspe, excluded_zeros = calculate_rmspe(y_true, preds)
    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = float(r2_score(y_true, preds))

    result = {
        "model_name": name,
        "rmspe": round(rmspe, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "train_time_sec": round(train_time_sec, 2),
        "inference_time_sec": round(inference_time_sec, 4),
        "rows_excluded_zero_sales": excluded_zeros,
    }

    logger.info(
        "[%s] RMSPE: %.4f | MAE: %.2f | RMSE: %.2f | R2: %.4f | Latency: %.4fs",
        name,
        rmspe,
        mae,
        rmse,
        r2,
        inference_time_sec,
    )
    return result


def find_latest_model_artifacts(models_dir: Path) -> Tuple[Dict[str, Path], Dict[str, float]]:
    """
    Identify model artifacts from manifest or directory inspection.
    """
    manifest_path = models_dir / "latest_manifest.json"
    model_paths: Dict[str, Path] = {}
    train_times: Dict[str, float] = {}

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for m_name, m_info in manifest.get("models", {}).items():
            f_path = Path(m_info.get("artifact_path", models_dir / m_info.get("filename", "")))
            if f_path.exists():
                model_paths[m_name] = f_path
                train_times[m_name] = float(m_info.get("train_time_sec", 0.0))

    if not model_paths:
        # Fallback to searching files
        for model_prefix in ["ridge", "xgboost", "mlp"]:
            matches = sorted(models_dir.glob(f"{model_prefix}_*.joblib"))
            if matches:
                latest_match = matches[-1]
                model_paths[model_prefix] = latest_match
                train_times[model_prefix] = 0.0

    return model_paths, train_times


def run_evaluation(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    out_path: str = "reports/comparison_metrics.json",
    config_path: str = "configs/rossmann_mapping.yaml",
) -> Dict[str, Any]:
    """
    Evaluate all serialized models on the holdout validation dataset and write metrics report.
    """
    models_p = Path(models_dir)
    data_p = Path(data_dir)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    val_file = data_p / "val_processed.parquet"
    if not val_file.exists():
        raise FileNotFoundError(f"Validation dataset not found at {val_file}")

    logger.info("Loading validation holdout data from %s", val_file)
    val_df = pd.read_parquet(val_file)

    mapping = load_column_mapping(config_path) if Path(config_path).exists() else None
    sales_col = mapping.sales if mapping else "Sales"

    if "Date" in val_df.columns:
        val_df = val_df.sort_values("Date").reset_index(drop=True)

    y_val = val_df[sales_col]
    feature_cols = [c for c in val_df.columns if c not in EXCLUDE_COLUMNS and c != sales_col]
    X_val = val_df[feature_cols]

    model_paths, train_times = find_latest_model_artifacts(models_p)
    if not model_paths:
        raise FileNotFoundError(f"No trained model artifacts found in {models_p}")

    model_results: List[Dict[str, Any]] = []

    for name, artifact_path in model_paths.items():
        logger.info("Loading %s artifact from %s", name, artifact_path)
        pipeline = joblib.load(artifact_path)
        res = evaluate_model_pipeline(
            name=name,
            pipeline=pipeline,
            X_val=X_val,
            y_val=y_val,
            train_time_sec=train_times.get(name, 0.0),
        )
        model_results.append(res)

    # Rank models by RMSPE ascending (lower is better)
    model_results = sorted(model_results, key=lambda m: m["rmspe"])

    report = {
        "validation_rows": int(len(val_df)),
        "generated_at": datetime.utcnow().isoformat(),
        "models": model_results,
    }

    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Consolidated comparison metrics saved -> %s", out_p)

    # Print formatted ranked summary table
    print("\n" + "=" * 90)
    print("                    FORECASTING BENCHMARK EVALUATION RESULTS")
    print("=" * 90)
    print(
        f"{'Rank':<5} {'Model':<12} {'RMSPE':<10} {'MAE':<10} {'RMSE':<10} {'R2':<8} {'Train (s)':<11} {'Infer (s)':<10}"
    )
    print("-" * 90)
    for rank, m in enumerate(model_results, start=1):
        print(
            f"#{rank:<4} {m['model_name']:<12} {m['rmspe']:<10.4f} {m['mae']:<10.2f} {m['rmse']:<10.2f} {m['r2']:<8.4f} {m['train_time_sec']:<11.2f} {m['inference_time_sec']:<10.4f}"
        )
    print("=" * 90)
    best_model = model_results[0]
    print(
        f">> Top Performing Model: {best_model['model_name'].upper()} with RMSPE = {best_model['rmspe']:.4f}\n"
    )

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 3b: Forecasting Model Evaluation")
    parser.add_argument("--data", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing serialized models")
    parser.add_argument("--out", type=str, default="reports/comparison_metrics.json", help="Path to output JSON report")
    parser.add_argument("--config", type=str, default="configs/rossmann_mapping.yaml", help="Path to column mapping config")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        data_dir=args.data,
        models_dir=args.models_dir,
        out_path=args.out,
        config_path=args.config,
    )
