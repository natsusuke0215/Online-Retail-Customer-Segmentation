"""Create development and final RFM snapshots with separate 90-day outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_segmentation.temporal import run_temporal_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_cleaned")
    parser.add_argument("--source", type=Path, default=ROOT / "raw_data" / "Online Retail.xlsx")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "temporal")
    args = parser.parse_args()
    summaries = run_temporal_pipeline(
        args.data_dir / "transactions_clean.csv",
        args.data_dir / "quality_report.json",
        args.source,
        args.data_dir / "temporal",
        args.report_dir,
    )
    for name, summary in summaries.items():
        print(
            f"{name}: {summary['observation_customers']:,} observed customers; "
            f"{summary['repurchased_observed_customers']:,} repurchased in 90 days; "
            f"{summary['future_new_customers_not_scored']:,} new future customers"
        )
    print(f"Features: {args.data_dir / 'temporal'}")


if __name__ == "__main__":
    main()
