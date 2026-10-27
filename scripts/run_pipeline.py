"""Run the baseline Online Retail cleaning, RFM, features, and EDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_segmentation.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "raw_data" / "Online Retail.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data_cleaned")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--snapshot-date", default="2011-12-10")
    args = parser.parse_args()
    config = PipelineConfig(args.source, args.output_dir, args.report_dir, args.snapshot_date)
    print(f"Reading {config.source} ...", flush=True)
    quality = run_pipeline(config)
    print(
        f"Done: {quality['eligible_rows']:,} eligible lines, "
        f"{quality['eligible_customers']:,} customers, "
        f"{quality['eligible_invoices']:,} invoices."
    )
    print(f"Data: {config.output_dir.resolve()}")
    print(f"EDA:  {config.report_dir.resolve()}")


if __name__ == "__main__":
    main()
