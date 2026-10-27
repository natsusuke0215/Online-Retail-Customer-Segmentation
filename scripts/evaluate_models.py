"""Score all submitted model labels on the same RFM feature matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_segmentation.evaluation import compare_model_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_cleaned")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "models" / "outputs")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    scores, profiles = compare_model_outputs(
        args.data_dir / "rfm_model_features.csv",
        args.data_dir / "rfm_customers.csv",
        args.predictions_dir,
        args.report_dir,
    )
    print(scores[["model", "n_clusters", "noise_fraction", "silhouette", "davies_bouldin", "calinski_harabasz", "status"]].to_string(index=False))
    print(f"Saved {len(profiles)} cluster profiles to {args.report_dir / 'cluster_profiles.csv'}")


if __name__ == "__main__":
    main()
