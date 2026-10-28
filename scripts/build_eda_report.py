"""Build a shareable EDA report from audited data and development outcomes only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "eda"
FIGURES = OUTPUT / "figures"
COLORS = ("#2563eb", "#14b8a6", "#f59e0b")


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quality(quality: dict) -> None:
    raw = quality["raw_rows"]
    values = [raw, raw - quality["raw_missing"]["CustomerID"], quality["eligible_rows"]]
    labels = ["Raw transaction lines", "Known CustomerID", "Eligible purchase lines"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bars = ax.barh(labels, values, color=["#94a3b8", COLORS[0], COLORS[1]])
    ax.invert_yaxis()
    ax.set(xlim=(0, raw * 1.18), xlabel="Number of lines", title="Rows available for customer-level RFM")
    for bar, value in zip(bars, values):
        ax.text(value + raw * .012, bar.get_y() + bar.get_height() / 2,
                f"{value:,} ({value / raw:.1%})", va="center")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "data_quality.png")


def plot_monthly(monthly: pd.DataFrame) -> None:
    months = monthly["Month"].astype(str).tolist()
    x = np.arange(len(months))
    colors = ["#94a3b8" if month == "2011-12" else COLORS[0] for month in months]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    axes[0].bar(x, monthly["eligible_lines"], color=colors)
    axes[0].set(ylabel="Eligible lines", title="Monthly purchase activity")
    axes[1].bar(x, monthly["eligible_revenue"] / 1e6, color=colors)
    axes[1].set(ylabel="Merchandise spend (£ million)", xlabel="Invoice month")
    axes[1].set_xticks(x, months, rotation=45, ha="right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].text(.99, .94, "Gray bar: Dec 2011 ends on day 9", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=9, color="#475569")
    save(fig, "monthly_activity.png")


def plot_rfm(rfm: pd.DataFrame) -> None:
    cols = ("Recency", "Frequency", "Monetary")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, col, color in zip(axes, cols, COLORS):
        ax.hist(np.log1p(rfm[col]), bins=35, color=color, edgecolor="white", linewidth=.4)
        ax.set(title=col, xlabel=f"log1p({col})", ylabel="Customers")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Development cohort: RFM before 1 June 2011", y=1.02)
    save(fig, "development_rfm.png")


def plot_relationships(rfm: pd.DataFrame) -> pd.DataFrame:
    cols = ["Recency", "Frequency", "Monetary"]
    corr = rfm[cols].corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].scatter(np.log1p(rfm["Frequency"]), np.log1p(rfm["Monetary"]),
                    s=9, alpha=.24, color=COLORS[0], rasterized=True)
    axes[0].set(xlabel="log1p(Frequency)", ylabel="log1p(Monetary)",
                title="Purchase count vs spend")
    heat = axes[1].imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_xticks(range(3), cols, rotation=30, ha="right")
    axes[1].set_yticks(range(3), cols)
    axes[1].set_title("Spearman correlation")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                         color="white" if abs(corr.iloc[i, j]) > .55 else "#0f172a")
    fig.colorbar(heat, ax=axes[1], fraction=.046, pad=.04)
    axes[0].spines[["top", "right"]].set_visible(False)
    save(fig, "development_relationships.png")
    return corr


def plot_outliers(rfm: pd.DataFrame) -> dict[str, tuple[float, int]]:
    cols = ("Recency", "Frequency", "Monetary")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    flagged = {}
    for ax, col, color in zip(axes, cols, COLORS):
        q1, q3 = rfm[col].quantile([.25, .75])
        upper = q3 + 1.5 * (q3 - q1)
        count = int(rfm[col].gt(upper).sum())
        flagged[col] = (float(upper), count)
        ax.boxplot(rfm[col], vert=False, patch_artist=True,
                   boxprops={"facecolor": color, "alpha": .7},
                   medianprops={"color": "#0f172a", "linewidth": 2},
                   flierprops={"marker": ".", "markersize": 3, "alpha": .3})
        ax.set_xscale("log")
        ax.set(title=f"{col} ({count} high flags)", xlabel=f"{col}, log axis")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Development RFM: IQR flags on original values", y=1.03)
    save(fig, "development_outliers.png")
    return flagged


def plot_repurchase(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = dev.copy()
    dev["recency_band"] = pd.cut(dev["Recency"], [0, 30, 90, 180, np.inf],
                                 labels=["1-30", "31-90", "91-180", "181+"], include_lowest=True)
    dev["frequency_band"] = pd.cut(dev["Frequency"], [0, 1, 2, 5, 10, np.inf],
                                   labels=["1", "2", "3-5", "6-10", "11+"], include_lowest=True)
    groups = []
    for col in ("recency_band", "frequency_band"):
        groups.append(dev.groupby(col, observed=True).agg(
            customers=("CustomerID", "size"),
            repurchase_rate=("repurchased_in_90d", "mean"),
        ))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    for ax, group, color, xlabel in zip(
        axes, groups, (COLORS[0], COLORS[1]), ("Recency (days)", "Frequency (invoices)")
    ):
        bars = ax.bar(group.index.astype(str), group["repurchase_rate"] * 100, color=color)
        ax.set(xlabel=xlabel, ylabel="Repurchased in next 90 days (%)", ylim=(0, 111))
        for bar, (_, row) in zip(bars, group.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{row['repurchase_rate']:.1%}\nn={int(row['customers']):,}",
                    ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Development cohort: descriptive repurchase rates", y=1.02)
    save(fig, "development_repurchase.png")
    return groups[0], groups[1]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    quality = json.loads((ROOT / "data_cleaned" / "quality_report.json").read_text(encoding="utf-8"))
    monthly = pd.read_csv(ROOT / "reports" / "tables" / "monthly_summary.csv")
    rfm = pd.read_csv(ROOT / "data_cleaned" / "temporal" / "development" / "rfm_customers.csv",
                      dtype={"CustomerID": "string"})
    outcomes = pd.read_csv(ROOT / "data_cleaned" / "temporal" / "outcomes" / "development.csv",
                           dtype={"CustomerID": "string"})
    dev = rfm.merge(outcomes, on="CustomerID", validate="one_to_one")
    if len(dev) != len(rfm) or set(dev["CustomerID"]) != set(outcomes["CustomerID"]):
        raise ValueError("Development RFM and outcomes do not have identical customers")

    plot_quality(quality)
    plot_monthly(monthly)
    plot_rfm(rfm)
    corr = plot_relationships(rfm)
    outliers = plot_outliers(rfm)
    recency, frequency = plot_repurchase(dev)
    top_n = max(1, int(np.ceil(len(rfm) * .01)))
    top_share = rfm["Monetary"].nlargest(top_n).sum() / rfm["Monetary"].sum()
    eligible_share = quality["eligible_rows"] / quality["raw_rows"]
    nov = monthly.loc[monthly["Month"].eq("2011-11")].iloc[0]
    report = f"""# Exploratory data analysis — Online Retail

Các hình dưới đây được tạo lại bằng `python scripts/build_eda_report.py` sau hai pipeline dữ liệu. **Kiểm tra chất lượng và chuỗi tháng** dùng toàn bộ nguồn để hiểu dữ liệu. **Quan hệ giữa RFM quá khứ và mua lại** chỉ dùng tập development: giao dịch trước 01/06/2011, kết quả trong `[2011-06-01, 2011-08-30)`. Không dùng outcome final để khám phá hay chọn model.

## 1. Chất lượng nguồn

![Số dòng qua các bước lọc](figures/data_quality.png)

Nguồn có **{quality['raw_rows']:,}** dòng; **{quality['raw_missing']['CustomerID']:,}** dòng thiếu `CustomerID` (**{quality['raw_missing']['CustomerID']/quality['raw_rows']:.1%}**) nên không thể gán vào RFM cá nhân. Còn **{quality['eligible_rows']:,}** dòng mua hàng hợp lệ (**{eligible_share:.1%}**). Có **{quality['normalized_exact_duplicates']:,}** dòng trùng hoàn toàn. Các cờ loại trừ có thể chồng nhau nên không cộng riêng từng cờ để tính số dòng bị loại. Chi tiết nằm trong `data_cleaned/quality_report.json` khi chạy pipeline.

## 2. Hoạt động theo tháng

![Hoạt động theo tháng](figures/monthly_activity.png)

Tháng 11/2011 có **{int(nov['eligible_lines']):,}** dòng mua hợp lệ và **£{nov['eligible_revenue']:,.0f}** giá trị mua hàng, cao nhất trong các tháng đầy đủ của nguồn. Đây là biến động quan sát được, chưa đủ để kết luận mùa vụ lặp lại vì dữ liệu chỉ dài khoảng một năm. Tháng 12/2011 chỉ có đến ngày 09/12 nên không so trực tiếp với tháng đầy đủ.

## 3. Phân phối RFM trước khi tạo cụm

![Phân phối RFM development](figures/development_rfm.png)

Tập development có **{len(rfm):,}** khách. Median `Recency/Frequency/Monetary` là **{rfm['Recency'].median():.0f} ngày / {rfm['Frequency'].median():.0f} hóa đơn / £{rfm['Monetary'].median():,.2f}**. Có **{(rfm['Frequency'].eq(1)).mean():.1%}** khách chỉ có một hóa đơn; **{top_n} khách (~1%)** có Monetary cao nhất chiếm **{top_share:.1%}** tổng Monetary quá khứ. Phân phối lệch mạnh nên feature dùng `log1p` trước khi fit `StandardScaler` trên development; các khách chi tiêu cao được giữ để kiểm tra độ nhạy, không xóa chỉ vì là ngoại lệ.

![Quan hệ giữa các biến RFM](figures/development_relationships.png)

`Frequency` và `Monetary` có tương quan Spearman **{corr.loc['Frequency','Monetary']:.2f}**; hai biến cùng mang tín hiệu về mức độ mua sắm. Điều này cần được lưu ý khi diễn giải khoảng cách và trọng số feature, nhưng chưa đủ lý do để bỏ một biến khỏi baseline chung của bốn model.

![Boxplot và cờ ngoại lệ development](figures/development_outliers.png)

IQR trên giá trị gốc gắn cờ **{outliers['Frequency'][1]}** khách có Frequency trên **{outliers['Frequency'][0]:.0f}** hóa đơn và **{outliers['Monetary'][1]}** khách có Monetary trên **£{outliers['Monetary'][0]:,.2f}**; Recency không có cờ phía cao. Trục boxplot được hiển thị log để nhìn được phần đuôi. Cờ ngoại lệ dùng để khảo sát độ nhạy, không phải nhãn gian lận hay quy tắc loại khách.

## 4. Quan hệ với mua lại 90 ngày trên development

![Tỷ lệ mua lại theo RFM quá khứ](figures/development_repurchase.png)

**{dev['repurchased_in_90d'].mean():.1%}** khách development mua lại trong 90 ngày. Nhóm mua gần nhất trong 1–30 ngày có tỷ lệ **{recency.loc['1-30','repurchase_rate']:.1%}** (`n={int(recency.loc['1-30','customers']):,}`), so với **{recency.loc['181+','repurchase_rate']:.1%}** ở nhóm hơn 180 ngày (`n={int(recency.loc['181+','customers']):,}`). Theo Frequency, khách một hóa đơn mua lại **{frequency.loc['1','repurchase_rate']:.1%}** (`n={int(frequency.loc['1','customers']):,}`), còn nhóm 11+ hóa đơn là **{frequency.loc['11+','repurchase_rate']:.1%}** (`n={int(frequency.loc['11+','customers']):,}`). Nhóm nhỏ có tỷ lệ kém ổn định; đây là **liên hệ mô tả**, không phải hiệu quả nhân quả của việc chăm sóc khách.

## Kết luận cho bước modeling

- Chỉ dùng `data_cleaned/temporal/development/rfm_model_features.csv` để chọn feature và siêu tham số. Ba cột `_scaled` là đầu vào; `CustomerID` là khóa.
- Giữ `final` và outcome 90 ngày riêng cho đánh giá sau khi chốt cách làm. Việc Frequency và Monetary tương quan, cùng doanh thu tập trung ở ít khách, là lý do cần so model với quy tắc Monetary/RFM đơn giản ở cùng số khách được chọn.
- Chưa có nhãn cụm, nên EDA này **chưa** chứng minh có bao nhiêu cụm hay thuật toán nào tốt hơn. Xem [hướng dẫn bàn giao](../team_data_guide.md).
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'README.md'} and 6 figures")


if __name__ == "__main__":
    main()
