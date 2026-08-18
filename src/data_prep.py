"""
data_prep.py
============
Module 1: Data Engineering & Preprocessing.

Responsibilities:
    - Ingest raw retail sales + store metadata (dataset-agnostic via column mapping).
    - Clean and impute missing values.
    - Engineer temporal features: calendar fields, lag features, rolling statistics.
    - Enforce a strict chronological (leak-free) train/validation split.

Design principles (non-negotiable, see project README):
    - No random splits on time-series data.
    - Lag/rolling windows only reference the past (t-1, t-7, ...).
    - CompetitionDistance -> median imputation.
    - Promo2Since* / PromoInterval -> zero-flag imputation (absence is meaningful).
    - Open == 0 rows are dropped before feature engineering (not genuine demand).
    - Open == 1 & Sales == 0 rows are kept but flagged (possible data anomalies),
      not silently dropped or silently trusted.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Configuration: dataset-agnostic column mapping
# ----------------------------------------------------------------------------
@dataclass
class ColumnMapping:
    """
    Maps this pipeline's internal canonical field names to the column names
    found in a given raw dataset. Allows swapping in another retail time-series
    dataset (e.g. Walmart Recruiting) without rewriting core logic -- only a
    new ColumnMapping needs to be supplied.
    """
    date: str = "Date"
    store_id: str = "Store"
    sales: str = "Sales"
    customers: str = "Customers"
    open_flag: str = "Open"
    promo: str = "Promo"
    day_of_week: str = "DayOfWeek"
    state_holiday: str = "StateHoliday"
    school_holiday: str = "SchoolHoliday"

    # store.csv fields
    store_type: str = "StoreType"
    assortment: str = "Assortment"
    competition_distance: str = "CompetitionDistance"
    competition_open_month: str = "CompetitionOpenSinceMonth"
    competition_open_year: str = "CompetitionOpenSinceYear"
    promo2: str = "Promo2"
    promo2_since_week: str = "Promo2SinceWeek"
    promo2_since_year: str = "Promo2SinceYear"
    promo_interval: str = "PromoInterval"


@dataclass
class DataPrepConfig:
    lag_days: list[int] = field(default_factory=lambda: [7, 14, 21, 30])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 30])
    validation_weeks: int = 6
    columns: ColumnMapping = field(default_factory=ColumnMapping)


# ----------------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------------
def load_raw_data(raw_dir: str | Path, cols: ColumnMapping) -> pd.DataFrame:
    """
    Load and merge sales records with store metadata.

    Args:
        raw_dir: Directory containing train.csv and store.csv.
        cols: Column mapping for the dataset being ingested.

    Returns:
        Merged DataFrame, one row per (store, date).
    """
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "train.csv"
    store_path = raw_dir / "store.csv"

    if not train_path.exists() or not store_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and store.csv in {raw_dir}. "
            f"Found: {[p.name for p in raw_dir.glob('*.csv')]}"
        )

    logger.info("Loading %s and %s", train_path.name, store_path.name)
    train = pd.read_csv(train_path, parse_dates=[cols.date], low_memory=False)
    store = pd.read_csv(store_path)

    _validate_schema(train, store, cols)

    df = train.merge(store, on=cols.store_id, how="left", validate="many_to_one")
    logger.info("Merged dataset shape: %s", df.shape)
    return df


def _validate_schema(train: pd.DataFrame, store: pd.DataFrame, cols: ColumnMapping) -> None:
    """Fail fast if required columns are missing from the raw inputs."""
    required_train = [
        cols.store_id, cols.date, cols.sales, cols.open_flag,
        cols.promo, cols.day_of_week, cols.state_holiday, cols.school_holiday,
    ]
    required_store = [cols.store_id, cols.store_type, cols.assortment, cols.competition_distance]

    missing_train = [c for c in required_train if c not in train.columns]
    missing_store = [c for c in required_store if c not in store.columns]

    if missing_train:
        raise ValueError(f"train data is missing required columns: {missing_train}")
    if missing_store:
        raise ValueError(f"store data is missing required columns: {missing_store}")


# ----------------------------------------------------------------------------
# Cleaning & Imputation
# ----------------------------------------------------------------------------
def clean_and_impute(df: pd.DataFrame, cols: ColumnMapping) -> pd.DataFrame:
    """
    Apply the project's imputation policy and filter out non-informative rows.

    - Drop rows where the store was closed (Open == 0): a closed store is not
      a demand signal and would corrupt lag/rolling features if left in.
    - Flag (not drop) rows where Open == 1 but Sales == 0, since these may be
      legitimate zero-demand days or data quality issues -- downstream modules
      can decide whether to exclude them.
    - CompetitionDistance: median imputation.
    - CompetitionOpenSinceMonth/Year: missing means no recorded competition yet;
      impute with 0 for the "months since competition opened" downstream feature.
    - Promo2SinceWeek/Year & PromoInterval: missing is *always* tied to
      Promo2 == 0 in this schema -- zero-flag imputation, not median, since
      the absence of a promo program is itself meaningful information.
    """
    df = df.copy()
    n_before = len(df)

    # --- Filter closed stores ---
    df = df[df[cols.open_flag] == 1].copy()
    logger.info("Dropped %d closed-store rows (Open == 0)", n_before - len(df))

    # --- Flag anomalous zero-sales-while-open rows ---
    df["is_zero_sales_anomaly"] = (df[cols.sales] == 0).astype(int)
    n_anomaly = df["is_zero_sales_anomaly"].sum()
    if n_anomaly:
        logger.warning(
            "%d rows have Open == 1 but Sales == 0 -- flagged via "
            "'is_zero_sales_anomaly', not dropped", n_anomaly
        )

    # --- CompetitionDistance: median imputation ---
    median_dist = df[cols.competition_distance].median()
    n_missing_dist = df[cols.competition_distance].isna().sum()
    df[cols.competition_distance] = df[cols.competition_distance].fillna(median_dist)
    if n_missing_dist:
        logger.info(
            "Imputed %d missing %s values with median (%.1f)",
            n_missing_dist, cols.competition_distance, median_dist,
        )

    # --- CompetitionOpenSince{Month,Year}: sentinel/zero imputation ---
    df["has_competition_open_date"] = (
        df[cols.competition_open_month].notna() & df[cols.competition_open_year].notna()
    ).astype(int)
    df[cols.competition_open_month] = df[cols.competition_open_month].fillna(0)
    df[cols.competition_open_year] = df[cols.competition_open_year].fillna(0)

    # --- Promo2Since{Week,Year} / PromoInterval: zero-flag imputation ---
    df[cols.promo2_since_week] = df[cols.promo2_since_week].fillna(0)
    df[cols.promo2_since_year] = df[cols.promo2_since_year].fillna(0)
    df[cols.promo_interval] = df[cols.promo_interval].fillna("")

    # --- Encode StateHoliday consistently (dataset uses '0' str for "none") ---
    df[cols.state_holiday] = df[cols.state_holiday].astype(str).replace({"0.0": "0"})

    return df


# ----------------------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame, cols: ColumnMapping) -> pd.DataFrame:
    """Derive calendar-based features from the date column."""
    df = df.copy()
    dt = df[cols.date]
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Quarter"] = dt.dt.quarter
    df["WeekOfYear"] = dt.dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df[cols.day_of_week].isin([6, 7]).astype(int)
    df["IsPromo"] = df[cols.promo].astype(int)
    return df


def add_competition_and_promo_dynamics(df: pd.DataFrame, cols: ColumnMapping) -> pd.DataFrame:
    """
    Elapsed-time features:
      - months since competition opened (0 if unknown / not yet open)
      - weeks since the store entered its Promo2 recurring-promo program
    """
    df = df.copy()

    comp_open = pd.to_datetime(
        dict(
            year=df[cols.competition_open_year].replace(0, np.nan),
            month=df[cols.competition_open_month].replace(0, np.nan),
            day=1,
        ),
        errors="coerce",
    )
    months_since_competition = (
        (df[cols.date].dt.year - comp_open.dt.year) * 12
        + (df[cols.date].dt.month - comp_open.dt.month)
    )
    df["MonthsSinceCompetition"] = months_since_competition.clip(lower=0).fillna(0)

    promo2_year = df[cols.promo2_since_year].replace(0, np.nan)
    promo2_week = df[cols.promo2_since_week].replace(0, np.nan)
    
    # Format week with 2 digits for ISO format "%G%V%u"
    week_str = promo2_week.dropna().astype(int).astype(str).str.zfill(2)
    promo2_start = pd.to_datetime(
        promo2_year.dropna().astype(int).astype(str) + week_str + "1",
        format="%G%V%u",
        errors="coerce",
    )
    weeks_since_promo2 = (df[cols.date] - promo2_start).dt.days / 7
    df["WeeksSincePromo2"] = weeks_since_promo2.clip(lower=0).fillna(0)

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame, cols: ColumnMapping, config: DataPrepConfig
) -> pd.DataFrame:
    """
    Construct per-store lag features and rolling window statistics.

    Critical: all lag/rolling computations are grouped by store_id and sorted
    by date ascending *before* shifting, so no future information leaks into
    a given row and no cross-store contamination occurs.
    """
    df = df.sort_values([cols.store_id, cols.date]).copy()
    grouped = df.groupby(cols.store_id, group_keys=False)[cols.sales]

    for lag in config.lag_days:
        df[f"sales_lag_{lag}"] = grouped.shift(lag)

    for window in config.rolling_windows:
        # shift(1) first so the current day's own sales is never included
        # in its own rolling window (would leak the target into the feature).
        df[f"sales_rolling_mean_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"sales_rolling_std_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).std()
        )

    return df


def engineer_features(df: pd.DataFrame, cols: ColumnMapping, config: DataPrepConfig) -> pd.DataFrame:
    """Run the full feature engineering sequence."""
    df = add_calendar_features(df, cols)
    df = add_competition_and_promo_dynamics(df, cols)
    df = add_lag_and_rolling_features(df, cols, config)
    return df


# ----------------------------------------------------------------------------
# Chronological Split
# ----------------------------------------------------------------------------
def chronological_train_val_split(
    df: pd.DataFrame, cols: ColumnMapping, validation_weeks: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train/validation using a strict cutoff date: the final
    `validation_weeks` weeks (by calendar date, across the whole dataset)
    become the validation holdout. This is applied globally, not per-store,
    so every store's validation window aligns on the same real-world dates --
    matching how the model would actually be deployed and evaluated.

    Random splits are never used for time-series data in this pipeline.
    """
    max_date = df[cols.date].max()
    cutoff = max_date - pd.Timedelta(weeks=validation_weeks)

    train = df[df[cols.date] <= cutoff].copy()
    val = df[df[cols.date] > cutoff].copy()

    logger.info(
        "Chronological split at %s -> train: %d rows (%s to %s), "
        "val: %d rows (%s to %s)",
        cutoff.date(), len(train), train[cols.date].min().date(),
        train[cols.date].max().date(), len(val),
        val[cols.date].min().date() if len(val) else "n/a",
        val[cols.date].max().date() if len(val) else "n/a",
    )
    return train, val


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_pipeline(raw_dir: str | Path, out_dir: str | Path, config: DataPrepConfig | None = None) -> None:
    """Execute the full Module 1 pipeline and persist processed outputs."""
    config = config or DataPrepConfig()
    cols = config.columns
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_dir, cols)
    df = clean_and_impute(df, cols)
    df = engineer_features(df, cols, config)
    train, val = chronological_train_val_split(df, cols, config.validation_weeks)

    train_path = out_dir / "train_processed.parquet"
    val_path = out_dir / "val_processed.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    logger.info("Saved processed train set -> %s", train_path)
    logger.info("Saved processed validation set -> %s", val_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 1: Data Engineering & Preprocessing")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory with train.csv/store.csv")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--validation-weeks", type=int, default=6, help="Weeks held out for validation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = DataPrepConfig(validation_weeks=args.validation_weeks)
    run_pipeline(args.raw_dir, args.out_dir, cfg)
