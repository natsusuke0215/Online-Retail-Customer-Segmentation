from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_segmentation.evaluation import compare_model_outputs, evaluate_labels
from retail_segmentation.pipeline import build_features, build_rfm, prepare_transactions


def sample_transactions() -> pd.DataFrame:
    rows = [
        ("100", "A1", "Item", 2, "2011-12-01 10:00", 10.0, 1.0, "UK"),
        ("100", "B1", "Item 2", 1, "2011-12-01 10:00", 5.0, 1.0, "UK"),
        ("100", "B1", "Item 2", 1, "2011-12-01 10:00", 5.0, 1.0, "UK"),
        ("101", "POST", "Postage", 1, "2011-12-02 10:00", 3.0, 1.0, "UK"),
        ("C102", "A1", "Return", -1, "2011-12-03 10:00", 10.0, 1.0, "UK"),
        ("103", "A1", "Item", 1, "2011-12-04 10:00", 8.0, np.nan, "UK"),
        ("104", "A1", "Item", 3, "2011-12-05 10:00", 7.0, 2.0, "FR"),
        ("105", "A1", "Future", 1, "2011-12-10 10:00", 9.0, 2.0, "FR"),
        ("106", "23444", "Next Day Carriage", 1, "2011-12-06 10:00", 15.0, 2.0, "FR"),
    ]
    return pd.DataFrame(rows, columns=[
        "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "UnitPrice", "CustomerID", "Country",
    ])


def test_cleaning_and_rfm_exclude_invalid_without_imputing_customer() -> None:
    tx, quality = prepare_transactions(sample_transactions(), "2011-12-10")
    assert quality["raw_exact_duplicates"] == 1
    assert quality["eligible_rows"] == 3
    assert quality["eligible_customers"] == 2
    assert quality["sensitivity"]["keep_exact_duplicates"]["amount_sum"] == 51.0
    assert quality["sensitivity"]["keep_non_merchandise_codes"]["amount_sum"] == 64.0
    assert tx.loc[3, "is_non_merchandise"]
    assert tx.loc[4, "is_cancellation"]
    assert tx.loc[5, "missing_customer_id"]
    assert tx.loc[7, "invalid_or_future_date"]
    assert tx.loc[8, "is_non_merchandise"]
    rfm = build_rfm(tx, "2011-12-10").set_index("CustomerID")
    assert rfm.loc["1", "Frequency"] == 1
    assert rfm.loc["1", "Recency"] == 9
    assert rfm.loc["1", "Monetary"] == 25.0
    assert rfm.loc["2", "Monetary"] == 21.0
    features, preprocessor = build_features(rfm.reset_index())
    assert len(features) == 2
    assert np.isfinite(features.iloc[:, 1:].to_numpy()).all()
    assert preprocessor["input_columns"] == ["Recency", "Frequency", "Monetary"]


def test_evaluation_excludes_dbscan_noise_and_handles_invalid_cluster_count() -> None:
    x = np.array([[0, 0], [0, 1], [10, 10], [10, 11], [50, 50]], dtype=float)
    scored = evaluate_labels(x, [0, 0, 1, 1, -1], silhouette_max_samples=10)
    assert scored["assigned_customers"] == 4
    assert scored["noise_customers"] == 1
    assert scored["n_clusters"] == 2
    assert scored["status"] == "ok"
    assert np.isfinite(scored["silhouette"])
    assert evaluate_labels(x, [0, 0, 0, 0, -1])["status"].startswith("undefined")
    with pytest.raises(ValueError, match="matching row counts"):
        evaluate_labels(x, [0, 1])


def test_comparison_joins_by_customer_id_and_handles_noise(tmp_path: Path) -> None:
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    reports = tmp_path / "reports"
    data.mkdir()
    outputs.mkdir()
    ids = ["1", "2", "3", "4", "5"]
    pd.DataFrame({
        "CustomerID": ids,
        "Recency_scaled": [0, 0, 10, 10, 50],
        "Frequency_scaled": [0, 1, 10, 11, 50],
        "Monetary_scaled": [0, 1, 10, 11, 50],
    }).to_csv(data / "rfm_model_features.csv", index=False)
    pd.DataFrame({
        "CustomerID": ids,
        "Recency": [1, 2, 10, 11, 50],
        "Frequency": [2, 2, 4, 4, 1],
        "Monetary": [20, 30, 40, 50, 5],
    }).to_csv(data / "rfm_customers.csv", index=False)
    pd.DataFrame({"CustomerID": ids[::-1], "cluster": ["-1", "1", "1", "0", "0"]}).to_csv(
        outputs / "dbscan.csv", index=False
    )
    scores, profiles = compare_model_outputs(
        data / "rfm_model_features.csv", data / "rfm_customers.csv", outputs, reports
    )
    assert scores.loc[0, "noise_customers"] == 1
    assert scores.loc[0, "n_clusters"] == 2
    assert profiles.loc[profiles["cluster"] == "-1", "customers"].iloc[0] == 1
    assert (reports / "model_comparison.csv").exists()
