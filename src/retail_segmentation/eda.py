"""EDA tables, figures, and a concise data-driven report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .pipeline import RFM_COLUMNS


def write_eda(tx: pd.DataFrame, rfm: pd.DataFrame, quality: dict, report_dir: Path) -> None:
    figure_dir = report_dir / "figures"
    table_dir = report_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    tx = tx.copy()
    tx["Month"] = tx["InvoiceDate"].dt.to_period("M").astype("string")
    monthly = tx.groupby("Month", dropna=False).agg(
        raw_lines=("InvoiceNo", "size"),
        missing_id_lines=("missing_customer_id", "sum"),
        cancellation_lines=("is_cancellation", "sum"),
        eligible_lines=("included_in_rfm", "sum"),
    )
    eligible = tx.loc[tx["included_in_rfm"]]
    monthly["eligible_invoices"] = eligible.groupby("Month")["InvoiceNo"].nunique()
    monthly["eligible_revenue"] = eligible.groupby("Month")["Amount"].sum().round(2)
    monthly = monthly.fillna(0).reset_index()
    monthly.to_csv(table_dir / "monthly_summary.csv", index=False)
    monthly["missing_id_fraction"] = monthly["missing_id_lines"] / monthly["raw_lines"]
    monthly["cancellation_fraction"] = monthly["cancellation_lines"] / monthly["raw_lines"]
    monthly[["Month", "raw_lines", "missing_id_fraction", "cancellation_fraction"]].to_csv(
        table_dir / "quality_by_month.csv", index=False
    )

    missing_country = tx.groupby("Country", dropna=False).agg(
        lines=("InvoiceNo", "size"),
        missing_id_lines=("missing_customer_id", "sum"),
        cancellation_lines=("is_cancellation", "sum"),
    )
    missing_country["missing_id_fraction"] = missing_country["missing_id_lines"] / missing_country["lines"]
    missing_country.sort_values("lines", ascending=False).to_csv(table_dir / "quality_by_country.csv")
    tx.loc[tx["is_exact_duplicate"], ["source_row", "InvoiceNo", "StockCode", "Quantity", "UnitPrice", "CustomerID"]].head(100).to_csv(
        table_dir / "duplicate_examples.csv", index=False
    )

    country = eligible.groupby("Country").agg(
        lines=("InvoiceNo", "size"),
        customers=("CustomerID", "nunique"),
        invoices=("InvoiceNo", "nunique"),
        revenue=("Amount", "sum"),
    ).sort_values("revenue", ascending=False)
    country["revenue"] = country["revenue"].round(2)
    country.to_csv(table_dir / "country_summary.csv")

    summary = rfm[RFM_COLUMNS].describe(percentiles=[.25, .5, .75, .9, .99]).T
    q1 = rfm[RFM_COLUMNS].quantile(.25)
    q3 = rfm[RFM_COLUMNS].quantile(.75)
    upper = q3 + 1.5 * (q3 - q1)
    summary["iqr_high_flag_count"] = (rfm[RFM_COLUMNS] > upper).sum()
    summary.to_csv(table_dir / "rfm_summary.csv")
    outlier_flags = rfm[["CustomerID", "PrimaryCountry"] + RFM_COLUMNS].copy()
    for col in RFM_COLUMNS:
        outlier_flags[f"{col}_iqr_high"] = rfm[col] > upper[col]
    outlier_flags = outlier_flags.loc[
        outlier_flags[[f"{col}_iqr_high" for col in RFM_COLUMNS]].any(axis=1)
    ].sort_values("Monetary", ascending=False)
    outlier_flags.head(100).to_csv(table_dir / "outlier_customer_examples.csv", index=False)
    corr_raw = rfm[RFM_COLUMNS].corr(method="spearman")
    corr_log = np.log1p(rfm[RFM_COLUMNS]).corr(method="pearson")
    corr_raw.to_csv(table_dir / "rfm_spearman_correlations.csv")
    corr_log.to_csv(table_dir / "rfm_log_correlations.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    miss = pd.Series(quality["raw_missing"]).sort_values(ascending=False)
    sns.barplot(x=miss.index, y=miss.values / quality["raw_rows"] * 100, ax=ax)
    ax.set(ylabel="Missing rows (%)", xlabel="Column", title="Missing values in raw source")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figure_dir / "missing_values.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(len(monthly))
    axes[0].plot(x, monthly["raw_lines"], label="All lines")
    axes[0].plot(x, monthly["eligible_lines"], label="Eligible merchandise lines")
    axes[0].set_ylabel("Lines")
    axes[0].legend()
    axes[1].bar(x, monthly["eligible_revenue"])
    axes[1].set(ylabel="Eligible revenue", xlabel="Month")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(monthly["Month"], rotation=45, ha="right")
    fig.suptitle("Monthly retail activity")
    fig.tight_layout()
    fig.savefig(figure_dir / "monthly_activity.png", dpi=160)
    plt.close(fig)

    logged = np.log1p(rfm[RFM_COLUMNS])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, RFM_COLUMNS):
        sns.histplot(logged[col], bins=35, ax=ax)
        ax.set(xlabel=f"log1p({col})", ylabel="Customers", title=col)
    fig.tight_layout()
    fig.savefig(figure_dir / "rfm_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, RFM_COLUMNS):
        sns.boxplot(y=logged[col], ax=ax)
        ax.set(ylabel=f"log1p({col})", title=col)
    fig.tight_layout()
    fig.savefig(figure_dir / "rfm_boxplots.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(logged["Frequency"], logged["Monetary"], s=7, alpha=.25)
    axes[0].set(xlabel="log1p(Frequency)", ylabel="log1p(Monetary)")
    sns.heatmap(corr_log, annot=True, vmin=-1, vmax=1, cmap="vlag", ax=axes[1])
    axes[1].set_title("Pearson correlations after log1p")
    fig.tight_layout()
    fig.savefig(figure_dir / "rfm_relationships.png", dpi=160)
    plt.close(fig)

    top_one_pct = max(1, int(np.ceil(len(rfm) * .01)))
    top_share = rfm["Monetary"].nlargest(top_one_pct).sum() / rfm["Monetary"].sum()
    single_purchase = (rfm["Frequency"] == 1).mean()
    sensitivity = quality["sensitivity"]
    duplicate_delta = sensitivity["keep_exact_duplicates"]["amount_sum"] - sensitivity["baseline"]["amount_sum"]
    code_delta = sensitivity["keep_non_merchandise_codes"]["amount_sum"] - sensitivity["baseline"]["amount_sum"]
    uk_missing = missing_country.loc["United Kingdom", "missing_id_lines"] / quality["raw_missing"]["CustomerID"]
    lines = [
        "# Kết quả EDA baseline", "",
        f"Nguồn: `{quality['source_file']}`; snapshot: **{quality['snapshot_date']}**.", "",
        f"- Gốc: **{quality['raw_rows']:,}** dòng; hợp lệ cho RFM: **{quality['eligible_rows']:,}** dòng, **{len(rfm):,}** khách hàng, **{quality['eligible_invoices']:,}** hóa đơn.",
        f"- Thiếu CustomerID: **{quality['raw_missing']['CustomerID']:,}** dòng ({quality['raw_missing']['CustomerID']/quality['raw_rows']:.1%}). Không suy đoán ID còn thiếu.",
        f"- Trùng hoàn toàn sau chuẩn hóa: **{quality['normalized_exact_duplicates']:,}** dòng; số dòng trùng trước chuẩn hóa: **{quality['raw_exact_duplicates']:,}**.",
        f"- Giữ các dòng trùng sẽ tăng tổng Amount **{duplicate_delta:,.2f}**; giữ mã phí/điều chỉnh sẽ tăng **{code_delta:,.2f}** và thêm **{sensitivity['keep_non_merchandise_codes']['customers'] - sensitivity['baseline']['customers']}** khách hàng.",
        f"- **{uk_missing:.1%}** dòng thiếu CustomerID thuộc United Kingdom; bảng theo tháng/quốc gia nằm trong `reports/tables/`.",
        f"- Median R/F/M: **{rfm['Recency'].median():.0f} ngày / {rfm['Frequency'].median():.0f} hóa đơn / {rfm['Monetary'].median():,.2f}**.",
        f"- Khách chỉ có một hóa đơn: **{single_purchase:.1%}**; top 1% khách theo Monetary chiếm **{top_share:.1%}** tổng Monetary.",
        "- Cờ IQR phía cao (không tự động xóa): " + ", ".join(f"{col}={int(summary.loc[col, 'iqr_high_flag_count']):,}" for col in RFM_COLUMNS) + ".",
        "", "## Diễn giải và giới hạn", "",
        "- Frequency và Monetary có đuôi dài; dùng log1p rồi StandardScaler cho baseline, giữ RFM gốc để diễn giải cụm.",
        "- Giao dịch hủy, giá/số lượng không dương và mã phí/điều chỉnh được gắn cờ trong transactions_clean.csv.",
        "- Việc loại dòng trùng hoàn toàn có thể làm thấp doanh thu nếu các dòng giống nhau là giao dịch thật; cần kiểm tra độ nhạy.",
        "- Tháng đầu và cuối có thể không đầy đủ; không diễn giải mức giảm của tháng 12/2011 như cả tháng.",
        "- Không có nhãn mục tiêu; các chỉ số đánh giá là nội tại và cần đọc cùng quy mô, độ cân bằng và ý nghĩa từng cụm.",
    ]
    (report_dir / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
