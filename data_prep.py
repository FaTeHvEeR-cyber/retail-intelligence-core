"""
Root wrapper for data_prep module.
Allows running `python data_prep.py` directly from project root.
"""

from src.data_prep import (
    ColumnMapping,
    DataPrepConfig,
    load_raw_data,
    clean_and_impute,
    add_calendar_features,
    add_competition_and_promo_dynamics,
    add_lag_and_rolling_features,
    engineer_features,
    chronological_train_val_split,
    run_pipeline,
    _parse_args,
)

if __name__ == "__main__":
    args = _parse_args()
    cfg = DataPrepConfig(validation_weeks=args.validation_weeks)
    run_pipeline(args.raw_dir, args.out_dir, cfg)
