"""Out-of-time baselines and cluster outcome profiles with strict ID alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import compare_model_outputs
from .pipeline import FEATURE_COLUMNS


def load_phase(data_dir: Path, phase: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if phase not in {"development", "final"}:
        raise ValueError("phase must be development or final")
    phase_dir = data_dir / "temporal" / phase
    rfm = pd.read_csv(phase_dir / "rfm_customers.csv", dtype={"CustomerID": "string"})
    features = pd.read_csv(phase_dir / "rfm_model_features.csv", dtype={"CustomerID": "string"})
    outcomes = pd.read_csv(data_dir / "temporal" / "outcomes" / f"{phase}.csv", dtype={"CustomerID": "string"})
    for name, table in (("rfm", rfm), ("features", features), ("outcomes", outcomes)):
        if table["CustomerID"].isna().any() or table["CustomerID"].duplicated().any():
            raise ValueError(f"{name} has missing or duplicate CustomerID")
    if not set(rfm["CustomerID"]) == set(features["CustomerID"]) == set(outcomes["CustomerID"]):
        raise ValueError("RFM, feature, and outcome customer sets differ")
    if not set(FEATURE_COLUMNS) <= set(features.columns):
        raise ValueError("Shared features are incomplete")
    if any(col in features.columns for col in ("repurchased_in_90d", "future_revenue", "future_invoices")):
        raise ValueError("Future outcomes leaked into model features")
    outcomes = outcomes.set_index("CustomerID").loc[rfm["CustomerID"]].reset_index()
    return rfm, features, outcomes


def evaluate_simple_rules(
    rfm: pd.DataFrame, outcomes: pd.DataFrame, capacity_fraction: float = 0.15
) -> pd.DataFrame:
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    if len(rfm) != len(outcomes) or set(rfm["CustomerID"]) != set(outcomes["CustomerID"]):
        raise ValueError("RFM and outcome customers differ")
    joined = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]].merge(
        outcomes[["CustomerID", "repurchased_in_90d", "future_revenue"]],
        on="CustomerID", validate="one_to_one",
    )
    if joined["future_revenue"].isna().any():
        raise ValueError("Missing future revenue")
    n_select = max(1, int(round(len(joined) * capacity_fraction)))
    r_rank = joined["Recency"].rank(method="average", pct=True)
    f_rank = joined["Frequency"].rank(method="average", pct=True)
    m_rank = joined["Monetary"].rank(method="average", pct=True)
    joined["RFM_score"] = (
        6 - np.ceil(5 * r_rank) + np.ceil(5 * f_rank) + np.ceil(5 * m_rank)
    )
    rules = {
        "top_monetary": ["Monetary", "Frequency", "Recency", "CustomerID"],
        "rfm_score": ["RFM_score", "Monetary", "Frequency", "Recency", "CustomerID"],
    }
    results = []
    total_revenue = float(joined["future_revenue"].sum())
    for name, columns in rules.items():
        ascending = [col in {"Recency", "CustomerID"} for col in columns]
        chosen = joined.sort_values(columns, ascending=ascending, kind="stable").head(n_select)
        selected_revenue = float(chosen["future_revenue"].sum())
        results.append({
            "rule": name,
            "eligible_customers": len(joined),
            "selected_customers": n_select,
            "selected_fraction": n_select / len(joined),
            "selected_repurchase_rate": float(chosen["repurchased_in_90d"].mean()),
            "selected_future_revenue": round(selected_revenue, 2),
            "future_revenue_share": selected_revenue / total_revenue if total_revenue > 0 else np.nan,
        })
    return pd.DataFrame(results)


def evaluate_cluster_priority(
    rfm: pd.DataFrame,
    outcomes: pd.DataFrame,
    labels: pd.DataFrame,
    model_name: str,
    capacity_fraction: float = 0.15,
) -> dict:
    """Prioritize whole clusters by past median Monetary, filling the last by past RFM."""
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    if labels["CustomerID"].isna().any() or labels["CustomerID"].duplicated().any():
        raise ValueError("Model labels need one non-null CustomerID per customer")
    if labels["cluster"].isna().any() or set(labels["CustomerID"]) != set(rfm["CustomerID"]):
        raise ValueError("Model labels must cover all observed customers")
    joined = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]].merge(
        labels[["CustomerID", "cluster"]], on="CustomerID", validate="one_to_one"
    ).merge(
        outcomes[["CustomerID", "repurchased_in_90d", "future_revenue"]],
        on="CustomerID", validate="one_to_one",
    )
    if joined[["repurchased_in_90d", "future_revenue"]].isna().any().any():
        raise ValueError("Missing future outcomes")
    joined["cluster"] = joined["cluster"].astype("string")
    cluster_order = joined.groupby("cluster", dropna=False).agg(
        past_monetary_median=("Monetary", "median"),
        past_frequency_median=("Frequency", "median"),
        past_recency_median=("Recency", "median"),
    ).reset_index()
    cluster_order["is_noise"] = cluster_order["cluster"].isin(["-1", "-1.0"])
    cluster_order = cluster_order.sort_values(
        ["is_noise", "past_monetary_median", "past_frequency_median", "past_recency_median", "cluster"],
        ascending=[True, False, False, True, True], kind="stable",
    )
    n_select = max(1, int(round(len(joined) * capacity_fraction)))
    ordered = pd.concat([
        joined.loc[joined["cluster"].eq(cluster)].sort_values(
            ["Monetary", "Frequency", "Recency", "CustomerID"],
            ascending=[False, False, True, True], kind="stable",
        ) for cluster in cluster_order["cluster"]
    ], ignore_index=True)
    chosen = ordered.head(n_select)
    total_revenue = float(joined["future_revenue"].sum())
    selected_revenue = float(chosen["future_revenue"].sum())
    return {
        "rule": f"{model_name}_cluster_priority",
        "eligible_customers": len(joined),
        "selected_customers": n_select,
        "selected_fraction": n_select / len(joined),
        "selected_repurchase_rate": float(chosen["repurchased_in_90d"].mean()),
        "selected_future_revenue": round(selected_revenue, 2),
        "future_revenue_share": selected_revenue / total_revenue if total_revenue > 0 else np.nan,
    }


def evaluate_submitted_models(
    data_dir: Path, predictions_dir: Path, report_dir: Path, phase: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rfm, _, outcomes = load_phase(data_dir, phase)
    phase_dir = data_dir / "temporal" / phase
    scores, _ = compare_model_outputs(
        phase_dir / "rfm_model_features.csv",
        phase_dir / "rfm_customers.csv",
        predictions_dir,
        report_dir,
    )
    future_profiles = []
    policies = evaluate_simple_rules(rfm, outcomes).to_dict("records")
    for path in sorted(predictions_dir.glob("*.csv")):
        labels = pd.read_csv(path, dtype={"CustomerID": "string", "cluster": "string"})
        policies.append(evaluate_cluster_priority(rfm, outcomes, labels, path.stem))
        labeled = rfm[["CustomerID"]].merge(labels, on="CustomerID", validate="one_to_one")
        labeled = labeled.merge(outcomes, on="CustomerID", validate="one_to_one")
        grouped = labeled.groupby("cluster", dropna=False).agg(
            customers=("CustomerID", "size"),
            repurchased=("repurchased_in_90d", "sum"),
            repurchase_rate=("repurchased_in_90d", "mean"),
            future_revenue=("future_revenue", "sum"),
            future_revenue_per_customer=("future_revenue", "mean"),
        ).reset_index()
        grouped["future_revenue_share"] = grouped["future_revenue"] / labeled["future_revenue"].sum()
        grouped.insert(0, "model", path.stem)
        future_profiles.append(grouped)
    profiles = pd.concat(future_profiles, ignore_index=True)
    profiles.to_csv(report_dir / "future_cluster_profiles.csv", index=False)
    pd.DataFrame(policies).to_csv(report_dir / "cluster_policy_comparison.csv", index=False)
    return scores, profiles
