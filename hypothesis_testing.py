"""Root execution wrapper for hypothesis testing module."""
# pyrefly: ignore [missing-import]
from src.hypothesis_testing import _parse_args, _load, run_hypothesis_tests
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = _parse_args()
    data = _load(args.data)
    summary = run_hypothesis_tests(data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved hypothesis test results -> %s", out_path)
