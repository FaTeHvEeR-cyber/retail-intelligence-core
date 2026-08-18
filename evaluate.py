"""Root execution wrapper for model evaluation."""
from src.evaluate import _parse_args, run_evaluation

if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        data_dir=args.data,
        models_dir=args.models_dir,
        out_path=args.out,
        config_path=args.config,
    )
