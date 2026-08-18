"""Root execution wrapper for model training."""
from src.train_models import _parse_args, train_forecasting_models

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
