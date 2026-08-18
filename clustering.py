"""Root execution wrapper for clustering module."""
from src.clustering import _parse_args, _load, build_store_profiles, cluster_stores
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
