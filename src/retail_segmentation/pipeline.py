"""Auditable transaction cleaning and one-row-per-customer RFM features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SOURCE_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]
RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]
FEATURE_COLUMNS = ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]
NON_MERCHANDISE_CODES = {"POST", "C2", "DOT", "BANK CHARGES", "M", "23444"}


@dataclass(frozen=True)
class PipelineConfig:
    source: Path
    output_dir: Path
    report_dir: Path
    snapshot_date: str = "2011-12-10"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        data = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        data = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    data = data.rename(columns={
        "Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"
    })
    missing = set(SOURCE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    return data[SOURCE_COLUMNS].copy()


def normalize_source(data: pd.DataFrame) -> pd.DataFrame:
    clean = data.copy()
    clean.insert(0, "source_row", np.arange(2, len(clean) + 2))
    for col in ["InvoiceNo", "StockCode", "Description", "Country"]:
        clean[col] = clean[col].astype("string").str.strip()
        clean[col] = clean[col].str.replace(r"\s+", " ", regex=True)
        clean[col] = clean[col].replace("", pd.NA)
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"], errors="coerce")
    clean["Quantity"] = pd.to_numeric(clean["Quantity"], errors="coerce")
    clean["UnitPrice"] = pd.to_numeric(clean["UnitPrice"], errors="coerce")
    customer_numeric = pd.to_numeric(clean["CustomerID"], errors="coerce")
    non_integer = customer_numeric.notna() & customer_numeric.mod(1).ne(0)
    if non_integer.any():
        raise ValueError(f"Non-integer CustomerID in {int(non_integer.sum())} rows")
    clean["CustomerID"] = customer_numeric.astype("Int64").astype("string")
    return clean


def prepare_transactions(data: pd.DataFrame, snapshot_date: str) -> tuple[pd.DataFrame, dict]:
    snapshot = pd.Timestamp(snapshot_date)
    if snapshot != snapshot.normalize():
        raise ValueError("snapshot_date must be a date without a time component")
    original_duplicate_count = int(data.duplicated(subset=SOURCE_COLUMNS).sum())
    tx = normalize_source(data)
    tx["is_exact_duplicate"] = tx.duplicated(subset=SOURCE_COLUMNS)
    tx["missing_customer_id"] = tx["CustomerID"].isna()
    tx["is_cancellation"] = tx["InvoiceNo"].str.startswith("C", na=False)
    tx["nonpositive_quantity"] = tx["Quantity"].isna() | tx["Quantity"].le(0)
    tx["nonpositive_unit_price"] = tx["UnitPrice"].isna() | tx["UnitPrice"].le(0)
    tx["is_non_merchandise"] = tx["StockCode"].str.upper().isin(NON_MERCHANDISE_CODES)
    tx["invalid_or_future_date"] = tx["InvoiceDate"].isna() | tx["InvoiceDate"].ge(snapshot)
    flags = [
        "is_exact_duplicate", "missing_customer_id", "is_cancellation",
        "nonpositive_quantity", "nonpositive_unit_price",
        "is_non_merchandise", "invalid_or_future_date",
    ]
    tx["included_in_rfm"] = ~tx[flags].any(axis=1)
    reasons = pd.Series("", index=tx.index, dtype="string")
    for flag in flags:
        mask = tx[flag]
        reasons.loc[mask] = reasons.loc[mask].where(reasons.loc[mask].eq(""), reasons.loc[mask] + ";") + flag
    tx["exclusion_reason"] = reasons.mask(reasons.eq(""), "included")
    tx["Amount"] = (tx["Quantity"] * tx["UnitPrice"]).round(2)

    eligible = tx.loc[tx["included_in_rfm"]]
    if eligible.empty:
        raise ValueError("No eligible purchase rows after applying cleaning rules")
    quality = {
        "raw_rows": int(len(tx)),
        "raw_missing": {col: int(data[col].isna().sum()) for col in SOURCE_COLUMNS},
        "raw_exact_duplicates": original_duplicate_count,
        "normalized_exact_duplicates": int(tx["is_exact_duplicate"].sum()),
        "flag_counts_overlapping": {flag: int(tx[flag].sum()) for flag in flags},
        "eligible_rows": int(len(eligible)),
        "excluded_rows": int((~tx["included_in_rfm"]).sum()),
        "eligible_customers": int(eligible["CustomerID"].nunique()),
        "eligible_invoices": int(eligible["InvoiceNo"].nunique()),
        "eligible_amount_sum": round(float(eligible["Amount"].sum()), 2),
        "date_min": tx["InvoiceDate"].min().isoformat() if tx["InvoiceDate"].notna().any() else None,
        "date_max": tx["InvoiceDate"].max().isoformat() if tx["InvoiceDate"].notna().any() else None,
        "snapshot_date": snapshot_date,
        "non_merchandise_codes": sorted(NON_MERCHANDISE_CODES),
        "notes": [
            "Flag counts overlap; do not sum them to obtain excluded_rows.",
            "CustomerID is not imputed because identity cannot be inferred.",
            "Exact duplicates are excluded for baseline; inspect sensitivity before final interpretation.",
            "Cancellations and returns are excluded from baseline RFM, not deleted from audit data.",
        ],
    }
    without_duplicate_filter = ~tx[[flag for flag in flags if flag != "is_exact_duplicate"]].any(axis=1)
    without_code_filter = ~tx[[flag for flag in flags if flag != "is_non_merchandise"]].any(axis=1)
    quality["sensitivity"] = {
        "baseline": {
            "rows": int(len(eligible)),
            "customers": int(eligible["CustomerID"].nunique()),
            "amount_sum": round(float(eligible["Amount"].sum()), 2),
        },
        "keep_exact_duplicates": {
            "rows": int(without_duplicate_filter.sum()),
            "customers": int(tx.loc[without_duplicate_filter, "CustomerID"].nunique()),
            "amount_sum": round(float(tx.loc[without_duplicate_filter, "Amount"].sum()), 2),
        },
        "keep_non_merchandise_codes": {
            "rows": int(without_code_filter.sum()),
            "customers": int(tx.loc[without_code_filter, "CustomerID"].nunique()),
            "amount_sum": round(float(tx.loc[without_code_filter, "Amount"].sum()), 2),
        },
    }
    return tx, quality


def build_rfm(tx: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    eligible = tx.loc[tx["included_in_rfm"]].copy()
    grouped = eligible.groupby("CustomerID", sort=True)
    rfm = grouped.agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Amount", "sum"),
        LineCount=("InvoiceNo", "size"),
    )
    rfm["Recency"] = (
        pd.Timestamp(snapshot_date) - rfm["LastPurchase"].dt.normalize()
    ).dt.days
    rfm["Monetary"] = rfm["Monetary"].round(2)
    if (rfm[RFM_COLUMNS] <= 0).any().any():
        raise ValueError("RFM features must be strictly positive for log1p baseline")
    country_counts = eligible.groupby(["CustomerID", "Country"], dropna=False).size()
    country_counts = country_counts.rename("lines").reset_index()
    country_counts = country_counts.sort_values(
        ["CustomerID", "lines", "Country"], ascending=[True, False, True]
    )
    primary_country = country_counts.drop_duplicates("CustomerID").set_index("CustomerID")["Country"]
    rfm["PrimaryCountry"] = primary_country
    rfm = rfm.reset_index()
    return rfm[["CustomerID", "Recency", "Frequency", "Monetary", "LastPurchase", "LineCount", "PrimaryCountry"]]


def build_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logged = np.log1p(rfm[RFM_COLUMNS].to_numpy(dtype=float))
    scaler = StandardScaler()
    transformed = scaler.fit_transform(logged)
    features = pd.DataFrame(transformed, columns=FEATURE_COLUMNS)
    features.insert(0, "CustomerID", rfm["CustomerID"].to_numpy())
    preprocessor = {
        "input_columns": RFM_COLUMNS,
        "output_columns": FEATURE_COLUMNS,
        "transform": "log1p_then_standard_scaler",
        "scaler": scaler,
    }
    return features, preprocessor


def run_pipeline(config: PipelineConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    raw = read_source(config.source)
    tx, quality = prepare_transactions(raw, config.snapshot_date)
    rfm = build_rfm(tx, config.snapshot_date)
    features, preprocessor = build_features(rfm)
    quality["source_file"] = str(config.source.resolve())
    quality["source_sha256"] = _sha256(config.source)
    quality["rfm_customer_count"] = int(len(rfm))
    quality["rfm_summary"] = {
        col: {name: round(float(value), 2) for name, value in rfm[col].describe(
            percentiles=[.25, .5, .75, .9, .99]
        ).items()} for col in RFM_COLUMNS
    }
    tx.to_csv(config.output_dir / "transactions_clean.csv", index=False)
    rfm.to_csv(config.output_dir / "rfm_customers.csv", index=False)
    features.to_csv(config.output_dir / "rfm_model_features.csv", index=False)
    joblib.dump(preprocessor, config.output_dir / "preprocessor.joblib")
    (config.output_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from .eda import write_eda

    write_eda(tx, rfm, quality, config.report_dir)
    return quality
