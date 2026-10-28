"""Check out-of-time feature construction and outcome isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_segmentation.pipeline import FEATURE_COLUMNS, prepare_transactions  # noqa: E402
from retail_segmentation.temporal import SNAPSHOTS, SnapshotSpec, build_snapshot  # noqa: E402
from retail_segmentation.temporal_evaluation import (  # noqa: E402
    evaluate_simple_rules,
    evaluate_submitted_models,
)


def sample_raw() -> pd.DataFrame:
    rows = [
        ("100", "A", "Item", 1, "2011-08-01", 10.0, 1, "UK"),
        ("101", "A", "Item", 1, "2011-08-10", 20.0, 2, "UK"),
        ("102", "A", "Item", 1, "2011-09-02", 30.0, 1, "UK"),
        ("103", "A", "Item", 1, "2011-09-03", 40.0, 3, "UK"),
        ("104", "A", "Item", 1, "2011-11-29", 25.0, 2, "UK"),
        ("105", "A", "Item", 1, "2011-11-30", 999.0, 1, "UK"),
        ("C106", "A", "Return", -1, "2011-10-01", 5.0, 1, "UK"),
    ]
    return pd.DataFrame(rows, columns=[
        "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "UnitPrice", "CustomerID", "Country",
    ])


def test_snapshot_uses_only_past_and_exact_90_day_future() -> None:
    tx, _ = prepare_transactions(sample_raw(), "2011-12-10")
    spec = SnapshotSpec("final", "2011-09-01")
    rfm, features, preprocessor, outcomes, report = build_snapshot(tx, spec)
    assert spec.end == pd.Timestamp("2011-11-30")
    assert rfm["CustomerID"].tolist() == ["1", "2"]
    assert report["observation_rows"] == 2
    assert report["future_rows"] == 3
    assert report["future_new_customers_not_scored"] == 1
    assert report["repurchased_observed_customers"] == 2
    assert outcomes["future_revenue"].tolist() == [30.0, 25.0]
    assert outcomes["future_invoices"].tolist() == [1, 1]
    assert list(features.columns) == ["CustomerID", *FEATURE_COLUMNS]
    assert not ({"future_revenue", "repurchased_in_90d"} & set(features.columns))
    assert np.allclose(
        preprocessor["scaler"].mean_,
        np.log1p(rfm[["Recency", "Frequency", "Monetary"]].to_numpy()).mean(axis=0),
    )

    changed = tx.copy()
    changed.loc[changed["InvoiceDate"].ge("2011-09-01"), "Amount"] *= 100
    _, changed_features, _, _, _ = build_snapshot(changed, spec)
    pd.testing.assert_frame_equal(features, changed_features)


def test_development_and_final_outcome_windows_do_not_overlap() -> None:
    development, final = SNAPSHOTS
    assert development.end <= pd.Timestamp(final.date)
    assert development.horizon_days == final.horizon_days == 90


def test_simple_rule_joins_shuffled_outcomes_by_customer_id() -> None:
    rfm = pd.DataFrame({
        "CustomerID": ["1", "2", "3", "4"],
        "Recency": [4, 3, 2, 1],
        "Frequency": [1, 2, 3, 4],
        "Monetary": [10, 20, 30, 40],
    })
    outcomes = pd.DataFrame({
        "CustomerID": ["4", "2", "1", "3"],
        "repurchased_in_90d": [1, 0, 0, 1],
        "future_revenue": [100.0, 0.0, 0.0, 10.0],
    })
    result = evaluate_simple_rules(rfm, outcomes, capacity_fraction=.5)
    top = result.set_index("rule").loc["top_monetary"]
    assert top["selected_customers"] == 2
    assert top["selected_repurchase_rate"] == 1.0
    assert top["selected_future_revenue"] == 110.0
    assert top["future_revenue_share"] == 1.0


def test_model_outcomes_join_by_id_after_teammate_submission(tmp_path: Path) -> None:
    phase_dir = tmp_path / "data" / "temporal" / "development"
    outcomes_dir = tmp_path / "data" / "temporal" / "outcomes"
    predictions_dir = tmp_path / "predictions"
    reports_dir = tmp_path / "reports"
    for path in (phase_dir, outcomes_dir, predictions_dir):
        path.mkdir(parents=True)
    rfm = pd.DataFrame({
        "CustomerID": ["1", "2", "3", "4"],
        "Recency": [1, 2, 10, 11],
        "Frequency": [5, 4, 1, 1],
        "Monetary": [50, 40, 10, 5],
    })
    rfm.to_csv(phase_dir / "rfm_customers.csv", index=False)
    pd.DataFrame({
        "CustomerID": ["1", "2", "3", "4"],
        "Recency_scaled": [0, 0.1, 5, 5.1],
        "Frequency_scaled": [0, 0.1, 5, 5.1],
        "Monetary_scaled": [0, 0.1, 5, 5.1],
    }).to_csv(phase_dir / "rfm_model_features.csv", index=False)
    pd.DataFrame({
        "CustomerID": ["4", "1", "3", "2"],
        "repurchased_in_90d": [0, 1, 0, 1],
        "future_invoices": [0, 1, 0, 1],
        "future_revenue": [0, 100, 0, 20],
    }).to_csv(outcomes_dir / "development.csv", index=False)
    pd.DataFrame({
        "CustomerID": ["3", "1", "4", "2"],
        "cluster": ["B", "A", "B", "A"],
    }).to_csv(predictions_dir / "kmeans.csv", index=False)
    scores, profiles = evaluate_submitted_models(
        tmp_path / "data", predictions_dir, reports_dir, "development"
    )
    assert scores.loc[0, "n_clusters"] == 2
    assert profiles.set_index("cluster").loc["A", "future_revenue"] == 120
    assert profiles.set_index("cluster").loc["B", "repurchased"] == 0
    assert (reports_dir / "future_cluster_profiles.csv").exists()
    policies = pd.read_csv(reports_dir / "cluster_policy_comparison.csv").set_index("rule")
    assert policies.loc["kmeans_cluster_priority", "selected_customers"] == 1
    assert policies.loc["kmeans_cluster_priority", "selected_future_revenue"] == 100
