"""Root execution wrapper for POS fraud detection module."""
# pyrefly: ignore [missing-import]
from src.fraud_detection import (
    ReturnSimConfig,
    AutoencoderConfig,
    run_pipeline,
    _parse_args,
)

if __name__ == "__main__":
    args = _parse_args()
    sim_cfg = ReturnSimConfig(
        n_transactions=args.n_transactions,
        anomaly_rate=args.anomaly_rate,
        random_seed=args.seed,
    )
    run_pipeline(
        sim_config=sim_cfg,
        ae_config=AutoencoderConfig(),
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )
