"""
hypothesis_testing.py
======================
Module 2a: Statistical Inference & Hypothesis Testing.

Responsibilities:
    - Welch's two-sample t-test: promo vs. non-promo daily sales.
    - One-Way ANOVA: sales variance across StoreType (and optionally Assortment).
    - Export test statistics, p-values, and an automated business interpretation.

Design principles:
    - Welch's t-test is used (not Student's) because equal variance between
      promo and non-promo groups cannot be safely assumed in retail data.
    - Significance level alpha = 0.05 throughout.
    - Operates on the cleaned Phase 1 output (Open == 0 rows already removed),
      so no closed-store zero-inflation biases the comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALPHA = 0.05


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": round(float(self.statistic), 4),
            "p_value": round(float(self.p_value), 6),
            "alpha": ALPHA,
            "significant": bool(self.significant),
            "interpretation": self.interpretation,
        }


def welch_ttest_promo_lift(
    df: pd.DataFrame, sales_col: str = "Sales", promo_col: str = "Promo"
) -> TestResult:
    """
    Two-sample Welch's t-test comparing mean daily sales during promotional
    periods (Promo == 1) against non-promotional periods (Promo == 0).

    Welch's t-test (equal_var=False) does not assume equal population
    variances between the two groups, which is the safer default for
    real-world retail sales distributions.
    """
    promo_sales = df.loc[df[promo_col] == 1, sales_col].dropna()
    non_promo_sales = df.loc[df[promo_col] == 0, sales_col].dropna()

    if len(promo_sales) == 0 or len(non_promo_sales) == 0:
        return TestResult(
            "welch_ttest_promo_vs_non_promo",
            0.0,
            1.0,
            False,
            "Insufficient data in promo or non-promo group to compute Welch's t-test.",
        )

    stat, p_value = stats.ttest_ind(promo_sales, non_promo_sales, equal_var=False)
    significant = bool(p_value < ALPHA)

    promo_mean = float(promo_sales.mean())
    non_promo_mean = float(non_promo_sales.mean())
    lift_pct = ((promo_mean - non_promo_mean) / non_promo_mean * 100) if non_promo_mean != 0 else 0.0

    if significant:
        interpretation = (
            f"Promotional periods show a statistically significant sales lift "
            f"of {lift_pct:.1f}% (promo mean={promo_mean:.0f} vs. "
            f"non-promo mean={non_promo_mean:.0f}, p={p_value:.2e}). "
            f"Reject H0: promotions are associated with higher sales."
        )
    else:
        interpretation = (
            f"No statistically significant difference detected between promo "
            f"(mean={promo_mean:.0f}) and non-promo (mean={non_promo_mean:.0f}) "
            f"sales (p={p_value:.4f}). Fail to reject H0."
        )

    logger.info("Welch's t-test: t=%.4f, p=%.6g, significant=%s", stat, p_value, significant)
    return TestResult("welch_ttest_promo_vs_non_promo", stat, p_value, significant, interpretation)


def anova_store_type(
    df: pd.DataFrame, sales_col: str = "Sales", group_col: str = "StoreType"
) -> TestResult:
    """
    One-Way ANOVA testing whether mean daily sales differ significantly
    across store type groups (A/B/C/D) or assortment levels.

    H0: all groups have the same mean sales.
    H1: at least one group's mean sales differs.
    """
    if group_col not in df.columns:
        return TestResult(
            f"anova_{group_col.lower()}",
            0.0,
            1.0,
            False,
            f"Column {group_col} not present in dataset.",
        )

    grouped_data = [g[sales_col].dropna().values for _, g in df.groupby(group_col)]
    group_labels = sorted(df[group_col].dropna().unique().tolist())

    if len(grouped_data) < 2:
        return TestResult(
            f"anova_{group_col.lower()}",
            0.0,
            1.0,
            False,
            f"Less than 2 groups found for column {group_col}.",
        )

    stat, p_value = stats.f_oneway(*grouped_data)
    significant = bool(p_value < ALPHA)

    group_means = df.groupby(group_col)[sales_col].mean().round(0).to_dict()

    if significant:
        interpretation = (
            f"Sales differ significantly across {group_col} groups "
            f"{group_labels} (F={stat:.4f}, p={p_value:.2e}). "
            f"Group means: {group_means}. Reject H0: {group_col} is "
            f"associated with meaningfully different sales performance."
        )
    else:
        interpretation = (
            f"No statistically significant difference in sales across "
            f"{group_col} groups {group_labels} (F={stat:.4f}, p={p_value:.4f}). "
            f"Fail to reject H0."
        )

    logger.info("ANOVA (%s): F=%.4f, p=%.6g, significant=%s", group_col, stat, p_value, significant)
    return TestResult(f"anova_{group_col.lower()}", stat, p_value, significant, interpretation)


def run_hypothesis_tests(df: pd.DataFrame) -> dict:
    """Run the full Module 2a test suite and return a JSON-serializable summary."""
    results = [
        welch_ttest_promo_lift(df),
        anova_store_type(df, group_col="StoreType"),
    ]
    if "Assortment" in df.columns:
        results.append(anova_store_type(df, group_col="Assortment"))

    return {"alpha": ALPHA, "n_rows_tested": len(df), "tests": [r.to_dict() for r in results]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 2a: Hypothesis Testing")
    parser.add_argument("--data", type=str, required=True, help="Path to processed train data (parquet or csv)")
    parser.add_argument("--out", type=str, default="reports/hypothesis_tests.json")
    return parser.parse_args()


def _load(path: str) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


if __name__ == "__main__":
    args = _parse_args()
    data = _load(args.data)
    summary = run_hypothesis_tests(data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved hypothesis test results -> %s", out_path)
