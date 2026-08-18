import numpy as np
import pandas as pd
import pytest

from src.clustering import build_store_profiles, determine_optimal_k, cluster_stores


@pytest.fixture
def multi_store_df():
    np.random.seed(42)
    records = []
    # 20 stores
    for store_id in range(1, 21):
        n_days = 100
        is_promo = np.random.choice([0, 1], size=n_days, p=[0.6, 0.4])
        base = 3000 + store_id * 300
        sales = base + is_promo * (1000 + store_id * 50) + np.random.normal(0, 200, size=n_days)
        customers = sales / 10 + np.random.normal(0, 20, size=n_days)
        for d in range(n_days):
            records.append({
                "Store": store_id,
                "Sales": max(0, sales[d]),
                "Customers": max(0, customers[d]),
                "Promo": is_promo[d],
            })
    return pd.DataFrame(records)


def test_build_store_profiles(multi_store_df):
    profiles = build_store_profiles(multi_store_df)
    assert len(profiles) == 20
    assert "avg_daily_sales" in profiles.columns
    assert "sales_cv" in profiles.columns
    assert "promo_lift_pct" in profiles.columns
    assert "promo_day_share" in profiles.columns
    assert (profiles["avg_daily_sales"] > 0).all()
    assert (profiles["promo_lift_pct"] > 0).all()


def test_cluster_stores(multi_store_df):
    profiles = build_store_profiles(multi_store_df)
    clustered, metadata = cluster_stores(profiles, k=3)
    assert len(clustered) == 20
    assert "cluster" in clustered.columns
    assert "pca_x" in clustered.columns
    assert "pca_y" in clustered.columns
    assert metadata["k"] == 3
    assert set(clustered["cluster"].unique()).issubset({0, 1, 2})
    assert len(metadata["pca_explained_variance_ratio"]) == 2
