"""Evaluate past-only selection rules; optionally score submitted cluster labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_segmentation.temporal_evaluation import (  # noqa: E402
    evaluate_simple_rules,
    evaluate_submitted_models,
    load_phase,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "final"], default="development")
    parser.add_argument("--with-models", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_cleaned")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "temporal")
    args = parser.parse_args()
    if args.phase == "final" and not args.with_models:
        parser.error("Final outcomes are reserved for the one-time comparison after model selection")
    rfm, _, outcomes = load_phase(args.data_dir, args.phase)
    output_dir = args.report_dir / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = evaluate_simple_rules(rfm, outcomes)
    baselines.to_csv(output_dir / "simple_rule_baselines.csv", index=False)
    print(baselines.to_string(index=False))
    if args.with_models:
        predictions_dir = args.predictions_dir or ROOT / "models" / "outputs" / args.phase
        scores, profiles = evaluate_submitted_models(
            args.data_dir, predictions_dir, output_dir, args.phase
        )
        print(f"Scored {len(scores)} models and {len(profiles)} future cluster profiles")


if __name__ == "__main__":
    main()
