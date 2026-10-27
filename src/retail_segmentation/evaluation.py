"""Common cluster metrics and comparison artifacts for team model outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .pipeline import FEATURE_COLUMNS, RFM_COLUMNS


def evaluate_labels(
    X: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    model_name: str = "model",
    runtime_seconds: float | None = None,
    silhouette_max_samples: int | None = None,
    random_state: int = 42,
) -> dict:
    """Score assigned customers; DBSCAN label -1 is noise and excluded from indices."""
    matrix = np.asarray(X, dtype=float)
    y = np.asarray(labels)
    if matrix.ndim != 2 or len(matrix) != len(y):
        raise ValueError("X and labels must have matching row counts")
    if not np.isfinite(matrix).all():
        raise ValueError("X contains missing or non-finite values")
    if pd.isna(y).any():
        raise ValueError("labels contain missing values")
    noise = pd.Series(y).astype("string").isin(["-1", "-1.0"]).to_numpy(dtype=bool)
    assigned = ~noise
    x_valid, y_valid = matrix[assigned], y[assigned]
    unique, counts = np.unique(y_valid, return_counts=True)
    result = {
        "model": model_name,
        "customers": int(len(y)),
        "assigned_customers": int(assigned.sum()),
        "noise_customers": int((~assigned).sum()),
        "noise_fraction": float((~assigned).mean()),
        "n_clusters": int(len(unique)),
        "cluster_sizes": {str(label): int(count) for label, count in zip(unique, counts)},
        "runtime_seconds": runtime_seconds,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "status": "ok",
    }
    if len(unique) < 2 or len(unique) >= len(y_valid):
        result["status"] = "undefined: need 2..n-1 non-noise clusters"
        return result
    result["davies_bouldin"] = float(davies_bouldin_score(x_valid, y_valid))
    result["calinski_harabasz"] = float(calinski_harabasz_score(x_valid, y_valid))
    if silhouette_max_samples is not None and len(y_valid) > silhouette_max_samples:
        rng = np.random.default_rng(random_state)
        chosen = []
        for label in unique:
            members = np.flatnonzero(y_valid == label)
            chosen.extend(rng.choice(members, size=min(2, len(members)), replace=False))
        chosen = np.unique(chosen)
        remaining = np.setdiff1d(np.arange(len(y_valid)), chosen, assume_unique=True)
        n_extra = max(0, min(silhouette_max_samples - len(chosen), len(remaining)))
        chosen = np.concatenate([chosen, rng.choice(remaining, size=n_extra, replace=False)])
        if len(np.unique(y_valid[chosen])) >= 2 and len(np.unique(y_valid[chosen])) < len(chosen):
            x_sil, y_sil = x_valid[chosen], y_valid[chosen]
        else:
            x_sil, y_sil = x_valid, y_valid
    else:
        x_sil, y_sil = x_valid, y_valid
    result["silhouette"] = float(silhouette_score(x_sil, y_sil))
    result["silhouette_customers"] = int(len(y_sil))
    return result


def compare_model_outputs(
    features_path: Path,
    rfm_path: Path,
    predictions_dir: Path,
    report_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(features_path, dtype={"CustomerID": "string"})
    rfm = pd.read_csv(rfm_path, dtype={"CustomerID": "string"})
    if features["CustomerID"].duplicated().any() or rfm["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique in shared feature and RFM tables")
    features = features.set_index("CustomerID")
    rfm = rfm.set_index("CustomerID")
    if set(features.index) != set(rfm.index):
        raise ValueError("Shared feature and RFM customer IDs differ")
    rfm = rfm.loc[features.index]
    metrics, profiles = [], []
    for path in sorted(predictions_dir.glob("*.csv")):
        predictions = pd.read_csv(path, dtype={"CustomerID": "string", "cluster": "string"})
        required = {"CustomerID", "cluster"}
        if not required <= set(predictions.columns):
            raise ValueError(f"{path.name} needs CustomerID and cluster columns")
        if predictions["CustomerID"].duplicated().any():
            raise ValueError(f"Duplicate CustomerID in {path.name}")
        predictions = predictions.set_index("CustomerID")
        if set(predictions.index) != set(features.index):
            raise ValueError(f"{path.name} does not cover exactly the shared customers")
        predictions = predictions.loc[features.index]
        if predictions["cluster"].isna().any():
            raise ValueError(f"Missing cluster label in {path.name}")
        labels = predictions["cluster"].to_numpy()
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        runtime = metadata.get("runtime_seconds")
        if runtime is not None and (not isinstance(runtime, (int, float)) or runtime < 0):
            raise ValueError(f"Invalid runtime_seconds in {metadata_path.name}")
        score = evaluate_labels(
            features[FEATURE_COLUMNS], labels, model_name=path.stem, runtime_seconds=runtime
        )
        score["parameters"] = json.dumps(metadata.get("parameters", {}), ensure_ascii=False, sort_keys=True)
        metrics.append(score)
        labeled = rfm[RFM_COLUMNS].copy()
        labeled["cluster"] = predictions["cluster"].to_numpy()
        grouped = labeled.groupby("cluster").agg(
            customers=("Recency", "size"),
            recency_median=("Recency", "median"),
            frequency_median=("Frequency", "median"),
            monetary_median=("Monetary", "median"),
            monetary_total=("Monetary", "sum"),
        ).reset_index()
        grouped.insert(0, "model", path.stem)
        profiles.append(grouped)
    if not metrics:
        raise FileNotFoundError(f"No model prediction CSV files found in {predictions_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(metrics)
    scores["cluster_sizes"] = scores["cluster_sizes"].map(str)
    profile = pd.concat(profiles, ignore_index=True)
    scores.to_csv(report_dir / "model_comparison.csv", index=False)
    profile.to_csv(report_dir / "cluster_profiles.csv", index=False)
    return scores, profile
