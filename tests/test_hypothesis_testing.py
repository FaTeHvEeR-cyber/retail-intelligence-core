import numpy as np
import pandas as pd
import pytest

from src.hypothesis_testing import welch_ttest_promo_lift, anova_store_type, run_hypothesis_tests


@pytest.fixture
def sample_retail_df():
    np.random.seed(42)
    n = 1000
    # StoreType: A, B, C, D
    store_types = np.random.choice(["a", "b", "c", "d"], size=n, p=[0.5, 0.1, 0.2, 0.2])
    assortments = np.random.choice(["a", "b", "c"], size=n)
    promos = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

    base_sales = {"a": 5000, "b": 10000, "c": 6000, "d": 7000}
    sales = [
        base_sales[st] + (2500 if promo == 1 else 0) + np.random.normal(0, 500)
        for st, promo in zip(store_types, promos)
    ]

    return pd.DataFrame({
        "Store": np.random.randint(1, 20, size=n),
        "Sales": sales,
        "Promo": promos,
        "StoreType": store_types,
        "Assortment": assortments,
    })


def test_welch_ttest_promo_lift(sample_retail_df):
    res = welch_ttest_promo_lift(sample_retail_df)
    assert res.significant is True
    assert res.p_value < 0.05
    assert res.statistic > 0
    assert "statistically significant" in res.interpretation


def test_anova_store_type(sample_retail_df):
    res = anova_store_type(sample_retail_df, group_col="StoreType")
    assert res.significant is True
    assert res.p_value < 0.05
    assert res.statistic > 0
    assert "differ significantly" in res.interpretation


def test_run_hypothesis_tests_summary(sample_retail_df):
    summary = run_hypothesis_tests(sample_retail_df)
    assert summary["alpha"] == 0.05
    assert summary["n_rows_tested"] == 1000
    assert len(summary["tests"]) >= 2
    for t in summary["tests"]:
        assert "test_name" in t
        assert "p_value" in t
        assert "interpretation" in t
