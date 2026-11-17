"""Check K-Means helpers on synthetic data; no Online Retail files required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import kmeans  # noqa: E402
from retail_segmentation.evaluation import compare_model_outputs  # noqa: E402

CENTERS = np.array([[-2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [2.0, -2.0, -2.0]])


def blobs() -> np.ndarray:
    X, _ = make_blobs(n_samples=300, centers=CENTERS, cluster_std=0.3, random_state=0)
    return X


def test_search_k_reports_every_k_with_finite_metrics() -> None:
    X = blobs()
    result = kmeans.search_k(X, (2, 3, 4), with_stability=False)
    assert result["k"].tolist() == [2, 3, 4]
    assert np.isfinite(result[["inertia", "silhouette", "davies_bouldin", "calinski_harabasz"]]).all().all()
    for sizes in result["cluster_sizes"]:
        assert sum(json.loads(sizes).values()) == len(X)
    assert result.loc[result["k"].eq(3), "silhouette"].iloc[0] == result["silhouette"].max()


def test_elbow_and_choice_find_three_separated_blobs() -> None:
    X = blobs()
    result = kmeans.search_k(X, tuple(range(2, 9)))
    elbow = kmeans.find_elbow_k(result["k"], result["inertia"])
    assert elbow == 3
    chosen, reason = kmeans.choose_k(result, elbow)
    assert chosen == 3
    assert "k=3" in reason
    assert result.loc[result["k"].eq(3), "ari_mean"].iloc[0] > 0.99


def test_relabel_orders_by_value_and_is_seed_independent() -> None:
    X = blobs()
    first = second = None
    for seed in (0, 7):
        model, labels, _ = kmeans.fit_kmeans(X, 3, random_state=seed)
        relabeled, centers = kmeans.relabel_by_value(model.cluster_centers_, labels)
        score = -centers[:, 0] + centers[:, 1] + centers[:, 2]
        assert np.all(np.diff(score) < 0)
        assert np.allclose(centers[0], CENTERS[0], atol=0.2)
        if first is None:
            first = relabeled
        else:
            second = relabeled
    assert np.array_equal(first, second)


def test_export_follows_team_contract(tmp_path: Path) -> None:
    X = blobs()[:40]
    ids = pd.Series([str(1000 + i) for i in range(len(X))], dtype="string")
    labels, _, runtime, _ = kmeans.fit_chosen_model(X, 3)
    csv_path, json_path = kmeans.export_labels(tmp_path / "outputs", "development", ids, labels, runtime, 3)
    exported = pd.read_csv(csv_path, dtype={"CustomerID": "string"})
    assert list(exported.columns) == ["CustomerID", "cluster"]
    assert exported["CustomerID"].tolist() == ids.tolist()
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["runtime_seconds"] >= 0
    assert metadata["parameters"]["n_clusters"] == 3

    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"CustomerID": ids, **dict(zip(kmeans.FEATURE_COLUMNS, X.T))}).to_csv(
        data / "rfm_model_features.csv", index=False
    )
    pd.DataFrame({
        "CustomerID": ids,
        "Recency": np.arange(1, len(X) + 1),
        "Frequency": np.ones(len(X), dtype=int),
        "Monetary": np.linspace(10, 100, len(X)),
    }).to_csv(data / "rfm_customers.csv", index=False)
    scores, _ = compare_model_outputs(
        data / "rfm_model_features.csv", data / "rfm_customers.csv", tmp_path / "outputs" / "development",
        tmp_path / "reports",
    )
    assert scores.loc[0, "model"] == "kmeans"
    assert scores.loc[0, "n_clusters"] == 3
    assert scores.loc[0, "status"] == "ok"


def test_cluster_names_follow_profile_not_label_number() -> None:
    rfm = pd.DataFrame({
        "CustomerID": [str(i) for i in range(6)],
        "Recency": [5, 6, 200, 210, 10, 220],
        "Frequency": [10, 12, 1, 1, 1, 6],
        "Monetary": [5000, 6000, 50, 60, 80, 3000],
        "PrimaryCountry": ["United Kingdom"] * 6,
    })
    labeled = rfm.assign(cluster=[2, 2, 0, 0, 1, 3])
    names = kmeans.name_clusters(kmeans.profile_clusters(labeled), rfm)
    assert "Trung thành" in names[2]
    assert len(set(names.values())) == len(names)
    assert names[0].endswith("Ngủ đông / mua ít")
    assert names[1].endswith("Khách mới / mua gần đây")
    assert names[3].endswith("Có nguy cơ rời bỏ (từng giá trị cao)")
