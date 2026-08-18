"""
clustering.py
=============
Module 2b: Store Behavior Clustering & Behavioral Profiling.

Responsibilities:
    - Aggregate daily records into store-level behavioral features.
    - Standardize features, determine optimal K via Elbow Method + Silhouette Score.
    - Fit K-Means and reduce to 2D via PCA for visualization.

Design principles:
    - Clustering operates on store-level aggregates, not raw daily rows.
      Clustering daily rows would group by day-of-week/seasonality noise
      rather than genuine store behavior.
    - Optimal K is chosen empirically (Silhouette Score), not hardcoded.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_store_profiles(
    df: pd.DataFrame,
    store_col: str = "Store",
    sales_col: str = "Sales",
    customers_col: str = "Customers",
    promo_col: str = "Promo",
) -> pd.DataFrame:
    """
    Aggregate daily sales records into one behavioral profile row per store.

    Engineered store-level features:
        - avg_daily_sales: mean daily sales (overall volume tier)
        - sales_cv: coefficient of variation (std/mean) -- sales volatility
        - avg_daily_customers: average daily foot traffic
        - promo_lift_pct: % increase in mean sales during promo vs. non-promo
          days, per store (behavioral responsiveness to promotions)
        - promo_day_share: fraction of a store's open days that were promo days
    """
    records = []
    for store_id, g in df.groupby(store_col):
        promo_sales = g.loc[g[promo_col] == 1, sales_col]
        non_promo_sales = g.loc[g[promo_col] == 0, sales_col]

        avg_sales = float(g[sales_col].mean())
        std_sales = float(g[sales_col].std()) if len(g) > 1 else 0.0
        cv = (std_sales / avg_sales) if avg_sales > 0 else 0.0

        if len(non_promo_sales) > 0 and non_promo_sales.mean() > 0:
            promo_lift_pct = float((promo_sales.mean() - non_promo_sales.mean()) / non_promo_sales.mean() * 100)
        else:
            promo_lift_pct = 0.0

        records.append(
            {
                store_col: store_id,
                "avg_daily_sales": avg_sales,
                "sales_cv": cv,
                "avg_daily_customers": float(g[customers_col].mean()) if customers_col in g.columns else 0.0,
                "promo_lift_pct": promo_lift_pct,
                "promo_day_share": float(g[promo_col].mean()) if promo_col in g.columns else 0.0,
                "n_open_days": int(len(g)),
            }
        )

    profiles = pd.DataFrame(records)

    # Sanity check: flag stores with abnormally few open-day records
    if len(profiles) > 0:
        low_data_threshold = profiles["n_open_days"].quantile(0.01)
        n_flagged = (profiles["n_open_days"] < low_data_threshold).sum()
        if n_flagged:
            logger.warning(
                "%d stores have unusually few open-day records (< %.0f days); "
                "their behavioral profiles may be unreliable.",
                n_flagged,
                low_data_threshold,
            )

        # Drop rows where critical metrics couldn't be computed
        n_before = len(profiles)
        profiles = profiles.dropna(subset=["avg_daily_sales", "sales_cv", "promo_lift_pct"])
        if len(profiles) < n_before:
            logger.warning("Dropped %d stores with insufficient data for clustering features.", n_before - len(profiles))

    return profiles.reset_index(drop=True)


def determine_optimal_k(
    X_scaled: np.ndarray, k_range: range = range(2, 11), random_state: int = 42
) -> tuple[int, dict]:
    """
    Determine optimal K via Elbow Method (inertia) and Silhouette Score.
    Returns the K with the highest silhouette score, plus the full score
    trace for reporting/visualization.
    """
    inertias = {}
    silhouette_scores = {}

    for k in k_range:
        if k >= len(X_scaled):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = float(km.inertia_)
        silhouette_scores[k] = float(silhouette_score(X_scaled, labels))

    if not silhouette_scores:
        return 2, {"inertias": {}, "silhouette_scores": {}}

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    logger.info(
        "Optimal K = %d (silhouette=%.4f). Full silhouette trace: %s",
        best_k,
        silhouette_scores[best_k],
        {k: round(v, 4) for k, v in silhouette_scores.items()},
    )
    return best_k, {"inertias": inertias, "silhouette_scores": silhouette_scores}


def cluster_stores(
    profiles: pd.DataFrame,
    feature_cols: list[str] | None = None,
    store_col: str = "Store",
    k: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Full clustering pipeline: scale -> determine K (if not provided) -> fit
    K-Means -> project to 2D via PCA.

    Returns:
        profiles with cluster labels + PCA coordinates attached,
        a metadata dict (chosen K, silhouette trace, explained variance).
    """
    if profiles.empty:
        raise ValueError("Profiles dataframe is empty. Cannot perform clustering.")

    feature_cols = feature_cols or [
        "avg_daily_sales",
        "sales_cv",
        "avg_daily_customers",
        "promo_lift_pct",
        "promo_day_share",
    ]

    # Filter to features that actually exist in profiles
    available_features = [c for c in feature_cols if c in profiles.columns]
    if not available_features:
        raise ValueError(f"None of {feature_cols} found in profiles dataframe.")

    X = profiles[available_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_selection_meta = {}
    if k is None:
        k, k_selection_meta = determine_optimal_k(X_scaled, random_state=random_state)

    # Ensure k <= number of samples
    k = min(k, len(profiles))

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    profiles = profiles.copy()
    profiles["cluster"] = km.fit_predict(X_scaled)

    n_pca_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    profiles["pca_x"] = coords[:, 0]
    profiles["pca_y"] = coords[:, 1] if n_pca_components > 1 else 0.0

    metadata = {
        "k": int(k),
        "feature_cols": available_features,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        **{key: {str(kk): vv for kk, vv in val.items()} for key, val in k_selection_meta.items()},
        "cluster_sizes": {int(k_id): int(cnt) for k_id, cnt in profiles["cluster"].value_counts().sort_index().items()},
        "cluster_profile_means": profiles.groupby("cluster")[available_features].mean().round(2).to_dict(orient="index"),
    }

    logger.info("K-Means fit with K=%d. Cluster sizes: %s", k, metadata["cluster_sizes"])
    return profiles, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 2b: Store Behavior Clustering")
    parser.add_argument("--data", type=str, required=True, help="Path to processed train data (parquet or csv)")
    parser.add_argument("--out", type=str, default="reports/store_clusters.json")
    parser.add_argument("--k", type=int, default=None, help="Force a specific K (skips auto-selection)")
    return parser.parse_args()


def _load(path: str) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


if __name__ == "__main__":
    args = _parse_args()
    data = _load(args.data)

    profiles = build_store_profiles(data)
    clustered, metadata = cluster_stores(profiles, k=args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    clustered_path = out_path.parent / "store_clusters.csv"
    clustered.to_csv(clustered_path, index=False)

    logger.info("Saved clustering metadata -> %s", out_path)
    logger.info("Saved per-store cluster assignments -> %s", clustered_path)
