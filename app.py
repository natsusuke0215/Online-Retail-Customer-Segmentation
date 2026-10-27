"""Streamlit integration demo for the shared RFM data and submitted clusters."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_cleaned"
OUTPUTS = ROOT / "models" / "outputs"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Online Retail | Customer Segmentation", layout="wide")
st.title("Online Retail · Phân cụm khách hàng")
st.caption("RFM thống nhất cho K-Means, Agglomerative, DBSCAN và GMM")

rfm_path = DATA / "rfm_customers.csv"
if not rfm_path.exists():
    st.info("Chạy `python scripts/run_pipeline.py` để tạo dữ liệu RFM trước.")
    st.stop()

rfm = pd.read_csv(rfm_path, dtype={"CustomerID": "string"})
left, middle, right = st.columns(3)
left.metric("Khách hàng", f"{len(rfm):,}")
middle.metric("Median Frequency", f"{rfm['Frequency'].median():.0f}")
right.metric("Median Monetary", f"{rfm['Monetary'].median():,.2f}")

with st.expander("Dữ liệu RFM và quy tắc"):
    st.write("Recency: ngày từ giao dịch cuối đến 2011-12-10. Frequency: số hóa đơn duy nhất. Monetary: tổng mua hàng hợp lệ.")
    st.dataframe(rfm.head(30), width="stretch")

comparison = REPORTS / "model_comparison.csv"
if comparison.exists():
    st.subheader("So sánh mô hình")
    scores = pd.read_csv(comparison)
    st.dataframe(scores, width="stretch")
    metric_tabs = st.tabs(["Silhouette", "Davies–Bouldin", "Calinski–Harabasz", "Noise"])
    for tab, metric in zip(metric_tabs, ["silhouette", "davies_bouldin", "calinski_harabasz", "noise_fraction"]):
        with tab:
            st.bar_chart(scores.set_index("model")[[metric]])

files = sorted(OUTPUTS.glob("*.csv")) if OUTPUTS.exists() else []
if not files:
    st.info("Chưa có nhãn cụm. Mỗi model xuất `models/outputs/<model>.csv` với cột CustomerID và cluster; sau đó chạy `python scripts/evaluate_models.py`.")
    st.stop()

selected = st.selectbox("Mô hình", files, format_func=lambda path: path.stem)
labels = pd.read_csv(selected, dtype={"CustomerID": "string", "cluster": "string"})
if labels["CustomerID"].duplicated().any() or set(labels["CustomerID"]) != set(rfm["CustomerID"]):
    st.error("File nhãn cần đúng một dòng cho mỗi CustomerID trong bảng RFM.")
    st.stop()
if labels["cluster"].isna().any():
    st.error("File nhãn có cluster rỗng.")
    st.stop()
labeled = rfm.merge(labels, on="CustomerID", validate="one_to_one")
st.subheader("Hồ sơ cụm")
profile = labeled.groupby("cluster", dropna=False).agg(
    customers=("CustomerID", "size"),
    recency_median=("Recency", "median"),
    frequency_median=("Frequency", "median"),
    monetary_median=("Monetary", "median"),
    monetary_total=("Monetary", "sum"),
).reset_index()
st.dataframe(profile, width="stretch")

st.subheader("Frequency × Monetary")
fig, ax = plt.subplots(figsize=(8, 5))
for cluster, part in labeled.groupby("cluster", dropna=False):
    ax.scatter(np.log1p(part["Frequency"]), np.log1p(part["Monetary"]), s=9, alpha=.45, label=str(cluster))
ax.set(xlabel="log1p(Frequency)", ylabel="log1p(Monetary)")
ax.legend(title="Cluster", markerscale=2, ncol=2)
st.pyplot(fig)
plt.close(fig)
