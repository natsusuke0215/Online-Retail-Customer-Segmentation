"""Out-of-time customer snapshots built from the audited transaction table."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import FEATURE_COLUMNS, RFM_COLUMNS, build_features, build_rfm


@dataclass(frozen=True)
class SnapshotSpec:
    name: str
    date: str
    horizon_days: int = 90

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.date) + pd.Timedelta(days=self.horizon_days)


SNAPSHOTS = (
    SnapshotSpec("development", "2011-06-01"),
    SnapshotSpec("final", "2011-09-01"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audited_transactions(path: Path) -> pd.DataFrame:
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Amount", "included_in_rfm"}
    tx = pd.read_csv(
        path,
        dtype={"CustomerID": "string", "InvoiceNo": "string"},
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )
    missing = required - set(tx.columns)
    if missing:
        raise ValueError(f"Audited transaction table lacks {sorted(missing)}")
    if not pd.api.types.is_bool_dtype(tx["included_in_rfm"]):
        raise ValueError("included_in_rfm must be boolean; rerun the baseline pipeline")
    if tx.loc[tx["included_in_rfm"], ["CustomerID", "InvoiceNo", "InvoiceDate", "Amount"]].isna().any().any():
        raise ValueError("Eligible transactions contain missing identifiers, dates, or amount")
    return tx


def build_snapshot(
    tx: pd.DataFrame, spec: SnapshotSpec
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict]:
    snapshot = pd.Timestamp(spec.date)
    if snapshot != snapshot.normalize() or spec.horizon_days <= 0:
        raise ValueError("Snapshot must be a date with a positive horizon")
    eligible = tx.loc[tx["included_in_rfm"]]
    observation = eligible.loc[eligible["InvoiceDate"].lt(snapshot)]
    future = eligible.loc[eligible["InvoiceDate"].ge(snapshot) & eligible["InvoiceDate"].lt(spec.end)]
    if observation.empty:
        raise ValueError(f"No observation transactions before {spec.date}")
    if future.empty:
        raise ValueError(f"No future transactions in {spec.name} horizon")

    rfm = build_rfm(observation, spec.date)
    features, preprocessor = build_features(rfm)
    future_by_customer = future.groupby("CustomerID", sort=True).agg(
        future_invoices=("InvoiceNo", "nunique"),
        future_revenue=("Amount", "sum"),
    )
    outcomes = pd.DataFrame({"CustomerID": rfm["CustomerID"]})
    outcomes = outcomes.join(future_by_customer, on="CustomerID")
    outcomes["future_invoices"] = outcomes["future_invoices"].fillna(0).astype(int)
    outcomes["future_revenue"] = outcomes["future_revenue"].fillna(0).round(2)
    outcomes["repurchased_in_90d"] = outcomes["future_invoices"].gt(0).astype(int)
    outcomes = outcomes[["CustomerID", "repurchased_in_90d", "future_invoices", "future_revenue"]]

    if outcomes["CustomerID"].tolist() != features["CustomerID"].tolist():
        raise AssertionError("Feature and outcome customer order differs")
    if not np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite feature value")
    observed_ids = set(rfm["CustomerID"])
    future_ids = set(future["CustomerID"])
    report = {
        "phase": spec.name,
        "snapshot_date": spec.date,
        "observation_end_exclusive": spec.date,
        "future_start_inclusive": spec.date,
        "future_end_exclusive": spec.end.strftime("%Y-%m-%d"),
        "horizon_days": spec.horizon_days,
        "observation_rows": int(len(observation)),
        "raw_observation_rows": int(tx["InvoiceDate"].lt(snapshot).sum()),
        "observation_missing_customer_id_rows": int(
            (tx["InvoiceDate"].lt(snapshot) & tx["missing_customer_id"]).sum()
        ),
        "observation_exact_duplicate_rows": int(
            (tx["InvoiceDate"].lt(snapshot) & tx["is_exact_duplicate"]).sum()
        ),
        "observation_customers": int(len(rfm)),
        "future_rows": int(len(future)),
        "future_customers": int(len(future_ids)),
        "future_new_customers_not_scored": int(len(future_ids - observed_ids)),
        "repurchased_observed_customers": int(outcomes["repurchased_in_90d"].sum()),
        "repurchase_rate": float(outcomes["repurchased_in_90d"].mean()),
        "future_revenue_observed_customers": round(float(outcomes["future_revenue"].sum()), 2),
        "rfm_median": {col: float(rfm[col].median()) for col in RFM_COLUMNS},
        "feature_columns": FEATURE_COLUMNS,
        "model_input_excludes": ["CustomerID", "repurchased_in_90d", "future_invoices", "future_revenue"],
        "preprocessing": "log1p(RFM), then StandardScaler fitted on observation customers only",
    }
    return rfm, features, preprocessor, outcomes, report


def run_temporal_pipeline(
    transactions_path: Path,
    baseline_quality_path: Path,
    raw_source_path: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, dict]:
    baseline_quality = json.loads(baseline_quality_path.read_text(encoding="utf-8"))
    if _sha256(raw_source_path) != baseline_quality.get("source_sha256"):
        raise ValueError("Raw source changed since baseline cleaning; rerun run_pipeline.py")
    if baseline_quality.get("snapshot_date") != "2011-12-10":
        raise ValueError("Temporal pipeline requires the full baseline audit through 2011-12-09")
    tx = read_audited_transactions(transactions_path)
    if len(tx) != baseline_quality.get("raw_rows"):
        raise ValueError("Audited transaction count differs from quality report")
    if int(tx["included_in_rfm"].sum()) != baseline_quality.get("eligible_rows"):
        raise ValueError("Audited eligible count differs from quality report")

    summaries = {}
    audit_hash = _sha256(transactions_path)
    for spec in SNAPSHOTS:
        rfm, features, preprocessor, outcomes, summary = build_snapshot(tx, spec)
        summary["audited_transactions_sha256"] = audit_hash
        summary["raw_source_sha256"] = baseline_quality["source_sha256"]
        phase_dir = output_dir / spec.name
        phase_dir.mkdir(parents=True, exist_ok=True)
        outcomes_dir = output_dir / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        phase_report_dir = report_dir / spec.name
        phase_report_dir.mkdir(parents=True, exist_ok=True)

        rfm.to_csv(phase_dir / "rfm_customers.csv", index=False)
        features.to_csv(phase_dir / "rfm_model_features.csv", index=False)
        joblib.dump(preprocessor, phase_dir / "preprocessor.joblib")
        outcomes.to_csv(outcomes_dir / f"{spec.name}.csv", index=False)
        (phase_dir / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rfm[RFM_COLUMNS].describe(percentiles=[.25, .5, .75, .9, .99]).to_csv(
            phase_report_dir / "rfm_summary.csv"
        )
        rfm[RFM_COLUMNS].corr(method="spearman").to_csv(
            phase_report_dir / "rfm_spearman_correlations.csv"
        )
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, col in zip(axes, RFM_COLUMNS):
            ax.hist(np.log1p(rfm[col]), bins=35, color="#4376a8", edgecolor="white")
            ax.set(title=col, xlabel=f"log1p({col})", ylabel="Customers")
        fig.tight_layout()
        fig.savefig(phase_report_dir / "rfm_distributions.png", dpi=140)
        plt.close(fig)
        (phase_report_dir / "data_summary.md").write_text(
            f"# {spec.name}: snapshot {spec.date}\n\n"
            f"Observation: {summary['observation_rows']:,} eligible rows, "
            f"{summary['observation_customers']:,} customers. "
            f"Future window: [{spec.date}, {summary['future_end_exclusive']}) "
            f"({spec.horizon_days} days).\n\n"
            f"Future transactions: {summary['future_rows']:,}; "
            f"new customers without past features: {summary['future_new_customers_not_scored']:,}.\n\n"
            "Only `rfm_model_features.csv` is model input. "
            "Outcome columns are stored separately and must not be used for clustering or preprocessing.\n",
            encoding="utf-8",
        )
        summaries[spec.name] = summary
    return summaries
