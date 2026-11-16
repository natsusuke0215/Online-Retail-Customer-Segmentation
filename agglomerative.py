"""Mô hình Phân cụm Phân cấp Tích lũy (Agglomerative Hierarchical Clustering)
cho bài toán Phân khúc Khách hàng Bán lẻ Trực tuyến (Online Retail Customer Segmentation).

Quy trình thực thi bao gồm:
1. Nạp dữ liệu giai đoạn Development (2.716 khách hàng) và Final (3.314 khách hàng).
2. Xây dựng ma trận liên kết phân cấp bằng scipy (các phương pháp: ward, complete, average, single).
3. Vẽ cây phả hệ Dendrogram, phân tích khoảng nhảy khoảng cách và các đường cắt ngang số cụm.
4. Tìm kiếm lưới (Grid Search) qua các tiêu chí liên kết (ward, complete, average) và số cụm k từ 2 đến 10.
5. Đánh giá đa chỉ số: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
6. Huấn luyện mô hình tối ưu và xuất tệp nhãn phân cụm theo hợp đồng dự án:
   - models/outputs/development/agglomerative.csv & .json
   - models/outputs/final/agglomerative.csv & .json
7. Phác họa chân dung khách hàng (Customer Persona Profiling): thống kê RFM gốc, phân tích PCA 2D, biểu đồ hộp Boxplot.
8. Đánh giá ngoài thời gian (Out-of-time Temporal Evaluation) trên hành vi mua lại trong 90 ngày tương lai.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Tự động định vị thư mục gốc chứa 'src/retail_segmentation'
def resolve_repo_root() -> Path:
    """Tự động xác định thư mục gốc của repository Online-Retail-Customer-Segmentation."""
    here = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    for candidate in [here, here.parent, here / "Online-Retail-Customer-Segmentation"]:
        if (candidate / "data_cleaned").exists() and (candidate / "src" / "retail_segmentation").exists():
            return candidate.resolve()
    return here.resolve()


ROOT = resolve_repo_root()
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from retail_segmentation.pipeline import FEATURE_COLUMNS, RFM_COLUMNS
    from retail_segmentation.temporal_evaluation import load_phase
except ImportError:
    # Dự phòng bổ sung cho IDE và môi trường chạy đặc thù
    sys.path.insert(0, str(Path("D:/WORKSPACE/Online-Retail-Customer-Segmentation/src").resolve()))
    from retail_segmentation.pipeline import FEATURE_COLUMNS, RFM_COLUMNS
    from retail_segmentation.temporal_evaluation import load_phase


def setup_plotting_style() -> None:
    """Thiết lập cấu hình giao diện đồ họa cho các báo cáo và biểu đồ."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "figure.autolayout": True,
        "figure.dpi": 300,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
    })


def create_directories() -> dict[str, Path]:
    """Khởi tạo và đảm bảo toàn bộ thư mục lưu báo cáo, hình ảnh và đầu ra tồn tại."""
    dirs = {
        "figures": ROOT / "reports" / "figures" / "agglomerative",
        "tables": ROOT / "reports" / "tables" / "agglomerative",
        "dev_out": ROOT / "models" / "outputs" / "development",
        "final_out": ROOT / "models" / "outputs" / "final",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def plot_dendrograms(X: np.ndarray, output_dir: Path) -> dict[str, float]:
    """Tính toán ma trận liên kết phân cấp và vẽ cây phả hệ Dendrogram với các ngưỡng cắt."""
    print("\n" + "=" * 70)
    print("BƯỚC 1: Tính toán ma trận liên kết phân cấp & Vẽ cây phả hệ Dendrogram")
    print("=" * 70)

    linkages = {
        "ward": linkage(X, method="ward", metric="euclidean"),
        "complete": linkage(X, method="complete", metric="euclidean"),
        "average": linkage(X, method="average", metric="euclidean"),
        "single": linkage(X, method="single", metric="euclidean"),
    }

    # Tên tiếng Việt của các phương pháp liên kết
    ten_phuong_phap = {
        "ward": "Ward (Tối thiểu hóa mức tăng phương sai nội cụm)",
        "complete": "Complete (Khoảng cách xa nhất giữa 2 phần tử)",
        "average": "Average (Khoảng cách trung bình giữa các cặp điểm)",
        "single": "Single (Khoảng cách gần nhất - Hiệu ứng xâu chuỗi)",
    }

    # 1. Vẽ so sánh 4 phương pháp liên kết (rút gọn hiển thị 30 nút gộp cấp cao nhất)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (method, Z) in enumerate(linkages.items()):
        ax = axes[idx]
        dendrogram(
            Z,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90,
            leaf_font_size=9,
            show_contracted=True,
            ax=ax,
        )
        ax.set_title(ten_phuong_phap[method], fontsize=11, fontweight="bold")
        ax.set_xlabel("Nhánh cụm rút gọn (Số trong ngoặc là số lượng khách hàng)")
        ax.set_ylabel("Khoảng cách gộp (Merger Distance)")

    plt.suptitle(
        "So Sánh Cây Phả Hệ Dendrogram Giữa 4 Phương Pháp Liên Kết (Giai Đoạn Development)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    comparison_path = output_dir / "dendrogram_comparison.png"
    plt.savefig(comparison_path)
    plt.close()
    print(f" -> Đã lưu biểu đồ so sánh Dendrogram: {comparison_path}")

    # 2. Cây phả hệ Ward Linkage chi tiết với phân tích đường cắt ngang
    Z_ward = linkages["ward"]
    fig, ax = plt.subplots(figsize=(12, 7))
    dendrogram(
        Z_ward,
        truncate_mode="lastp",
        p=35,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
        ax=ax,
        color_threshold=38.0,
    )

    # Ngưỡng cắt ngang ứng với k=4 và k=5
    cut_k4 = 38.0
    cut_k5 = 31.0

    ax.axhline(y=cut_k4, color="crimson", linestyle="--", linewidth=2, label=f"Đường cắt k=4 (Ngưỡng khoảng cách = {cut_k4})")
    ax.axhline(y=cut_k5, color="darkorange", linestyle=":", linewidth=2, label=f"Đường cắt k=5 (Ngưỡng khoảng cách = {cut_k5})")

    ax.set_title("Cây Phả Hệ Ward Linkage Với Các Đường Cắt Ngang Lựa Chọn Số Cụm k", fontsize=13, fontweight="bold")
    ax.set_xlabel("Các nhánh phân cụm (Số trong ngoặc biểu thị quy mô số lượng khách hàng)")
    ax.set_ylabel("Mức gia tăng phương sai Ward (ΔESS)")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    ward_dendro_path = output_dir / "dendrogram_ward.png"
    plt.savefig(ward_dendro_path)
    plt.close()
    print(f" -> Đã lưu biểu đồ chi tiết Dendrogram Ward: {ward_dendro_path}")

    return {
        "ward_cut_k4": cut_k4,
        "ward_cut_k5": cut_k5,
    }


def grid_search_hyperparameters(
    X: np.ndarray,
    k_range: range,
    linkage_methods: list[str],
    output_table_dir: Path,
    output_fig_dir: Path,
) -> pd.DataFrame:
    """Thực hiện tìm kiếm lưới (Grid Search) qua các phương pháp liên kết và số cụm k (2..10)."""
    print("\n" + "=" * 70)
    print("BƯỚC 2: Tìm kiếm lưới (Grid Search) qua các tiêu chí liên kết & Số cụm k từ 2 đến 10")
    print("=" * 70)

    records = []
    for method in linkage_methods:
        print(f"  Đang đánh giá phương pháp liên kết: '{method}'...")
        for k in k_range:
            t0 = time.perf_counter()
            model = AgglomerativeClustering(n_clusters=k, linkage=method, metric="euclidean")
            labels = model.fit_predict(X)
            elapsed = time.perf_counter() - t0

            unique, counts = np.unique(labels, return_counts=True)
            sil = float(silhouette_score(X, labels))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))

            min_size = int(np.min(counts))
            max_size = int(np.max(counts))

            records.append({
                "linkage": method,
                "k": k,
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "calinski_harabasz": round(ch, 2),
                "min_cluster_size": min_size,
                "max_cluster_size": max_size,
                "runtime_seconds": round(elapsed, 4),
            })

    results_df = pd.DataFrame(records)
    table_path = output_table_dir / "linkage_comparison.csv"
    results_df.to_csv(table_path, index=False)
    print(f" -> Đã lưu bảng kết quả tìm kiếm lưới: {table_path}")

    # Bảng hiển thị tiếng Việt dễ đọc trên terminal
    results_display = pd.DataFrame({
        "Phương pháp liên kết": results_df["linkage"].map({
            "ward": "Ward (Phương sai)",
            "complete": "Complete (Cực đại)",
            "average": "Average (Trung bình)",
        }),
        "Số cụm (k)": results_df["k"],
        "Silhouette": results_df["silhouette"],
        "Davies-Bouldin": results_df["davies_bouldin"],
        "Calinski-Harabasz": results_df["calinski_harabasz"],
        "Cụm nhỏ nhất": results_df["min_cluster_size"],
        "Cụm lớn nhất": results_df["max_cluster_size"],
        "Thời gian (giây)": results_df["runtime_seconds"],
    })
    results_vi_path = output_table_dir / "so_sanh_cac_phuong_phap_lien_ket.csv"
    results_display.to_csv(results_vi_path, index=False, encoding="utf-8-sig")
    print(f" -> Đã lưu bảng so sánh tiếng Việt: {results_vi_path}")
    print("\n--- BẢNG SO SÁNH CÁC PHƯƠNG PHÁP LIÊN KẾT (TRÍCH ĐOẠN ĐẦU) ---")
    print(results_display.head(12).to_string(index=False))

    # Vẽ biểu đồ so sánh 3 chỉ số theo số cụm k
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    palette = {"ward": "#1f77b4", "complete": "#2ca02c", "average": "#ff7f0e"}
    markers = {"ward": "o", "complete": "s", "average": "^"}
    labels_vi = {"ward": "Ward (Phương sai)", "complete": "Complete (Cực đại)", "average": "Average (Trung bình)"}

    # 1. Silhouette Score
    for method in linkage_methods:
        sub = results_df[results_df["linkage"] == method]
        axes[0].plot(sub["k"], sub["silhouette"], marker=markers[method], label=labels_vi[method], color=palette[method])
    axes[0].set_title("Hệ số Silhouette (Càng cao càng tốt)", fontweight="bold")
    axes[0].set_xlabel("Số lượng cụm (k)")
    axes[0].set_ylabel("Hệ số Silhouette")
    axes[0].legend()

    # 2. Davies-Bouldin Index
    for method in linkage_methods:
        sub = results_df[results_df["linkage"] == method]
        axes[1].plot(sub["k"], sub["davies_bouldin"], marker=markers[method], label=labels_vi[method], color=palette[method])
    axes[1].set_title("Chỉ số Davies-Bouldin (Càng thấp càng tốt)", fontweight="bold")
    axes[1].set_xlabel("Số lượng cụm (k)")
    axes[1].set_ylabel("Chỉ số Davies-Bouldin")
    axes[1].legend()

    # 3. Calinski-Harabasz Index
    for method in linkage_methods:
        sub = results_df[results_df["linkage"] == method]
        axes[2].plot(sub["k"], sub["calinski_harabasz"], marker=markers[method], label=labels_vi[method], color=palette[method])
    axes[2].set_title("Chỉ số Calinski-Harabasz (Càng cao càng tốt)", fontweight="bold")
    axes[2].set_xlabel("Số lượng cụm (k)")
    axes[2].set_ylabel("Chỉ số Calinski-Harabasz")
    axes[2].legend()

    plt.suptitle("So Sánh Các Chỉ Số Đo Lường Hiệu Quả Phân Cụm Theo Số Cụm k", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    metrics_path = output_fig_dir / "metrics_comparison.png"
    plt.savefig(metrics_path)
    plt.close()
    print(f" -> Đã lưu biểu đồ so sánh chỉ số đo lường: {metrics_path}")

    return results_df


def fit_and_export_model(
    phase: str,
    features_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    best_k: int,
    best_linkage: str,
    output_dir: Path,
) -> tuple[np.ndarray, dict]:
    """Huấn luyện mô hình Agglomerative tối ưu và xuất tệp nhãn phân cụm theo hợp đồng dự án."""
    X = features_df[FEATURE_COLUMNS].to_numpy()
    customer_ids = features_df["CustomerID"].to_numpy()

    t0 = time.perf_counter()
    model = AgglomerativeClustering(n_clusters=best_k, linkage=best_linkage, metric="euclidean")
    labels = model.fit_predict(X)
    runtime = time.perf_counter() - t0

    # Đảm bảo đúng định dạng bảng kết quả gồm 2 cột CustomerID và cluster
    output_df = pd.DataFrame({
        "CustomerID": customer_ids,
        "cluster": labels.astype(str),
    })

    csv_path = output_dir / "agglomerative.csv"
    output_df.to_csv(csv_path, index=False)

    metadata = {
        "model": "agglomerative",
        "phase": phase,
        "customers": len(labels),
        "runtime_seconds": round(runtime, 4),
        "parameters": {
            "n_clusters": best_k,
            "linkage": best_linkage,
            "metric": "euclidean",
        },
        "metrics": {
            "silhouette": round(float(silhouette_score(X, labels)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 2),
        },
    }

    json_path = output_dir / "agglomerative.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    phase_ten = "Giai đoạn Development" if phase == "development" else "Giai đoạn Final"
    print(f"  [{phase_ten.upper()}] Đã xuất các tệp mô hình:")
    print(f"     Tệp nhãn CSV:  {csv_path} ({len(output_df):,} khách hàng)")
    print(f"     Tệp siêu dữ liệu JSON: {json_path}")

    return labels, metadata


def generate_customer_profiles(
    rfm_df: pd.DataFrame,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    output_fig_dir: Path,
    output_table_dir: Path,
) -> pd.DataFrame:
    """Phác họa chân dung khách hàng, tính toán thống kê mô tả RFM thực tế và vẽ biểu đồ trực quan."""
    print("\n" + "=" * 70)
    print("BƯỚC 3: Phác họa chân dung khách hàng & Phân khúc kinh doanh thực tế")
    print("=" * 70)

    df = rfm_df.copy()
    df["cluster"] = labels

    # Tính toán hồ sơ thống kê mô tả RFM
    summary = df.groupby("cluster").agg(
        customers=("CustomerID", "count"),
        recency_median=("Recency", "median"),
        recency_mean=("Recency", "mean"),
        frequency_median=("Frequency", "median"),
        frequency_mean=("Frequency", "mean"),
        monetary_median=("Monetary", "median"),
        monetary_mean=("Monetary", "mean"),
        monetary_total=("Monetary", "sum"),
    ).reset_index()

    total_revenue = df["Monetary"].sum()
    summary["revenue_share"] = summary["monetary_total"] / total_revenue
    summary["customer_share"] = summary["customers"] / len(df)

    # Đặt tên tiếng Việt rõ ràng cho từng nhóm chân dung khách hàng
    # Sắp xếp các cụm theo trung vị Monetary giảm dần để gán tên chuẩn hóa
    summary_sorted = summary.sort_values(by="monetary_median", ascending=False).reset_index(drop=True)
    persona_mapping = {}
    names_vi = [
        "Khách hàng VIP / Quán quân (Chi tiêu khủng, Mua thường xuyên)",
        "Khách hàng trung thành (Gắn kết đều, Giá trị khá)",
        "Khách hàng tiềm năng (Mới mua gần đây, Đầy triển vọng)",
        "Khách hàng có nguy cơ rời bỏ / Ngủ đông (Lâu không mua, Tần suất thấp)",
        "Khách hàng mua một lần rồi rời đi",
    ]
    for i, row in summary_sorted.iterrows():
        c_id = row["cluster"]
        persona_name = names_vi[i] if i < len(names_vi) else f"Phân khúc {c_id}"
        persona_mapping[c_id] = persona_name

    summary["persona"] = summary["cluster"].map(persona_mapping)
    df["persona"] = df["cluster"].map(persona_mapping)

    # Sắp xếp lại thứ tự cột cho rõ ràng
    summary = summary[[
        "cluster", "persona", "customers", "customer_share",
        "recency_median", "frequency_median", "monetary_median",
        "recency_mean", "frequency_mean", "monetary_mean",
        "monetary_total", "revenue_share"
    ]]

    profile_table_path = output_table_dir / "cluster_profiles.csv"
    summary.to_csv(profile_table_path, index=False)
    print(f" -> Đã lưu bảng hồ sơ phân khúc (hợp đồng dự án): {profile_table_path}")

    # Bảng tiếng Việt định dạng đẹp mắt trên terminal
    summary_display = pd.DataFrame({
        "Mã cụm": summary["cluster"],
        "Chân dung khách hàng": summary["persona"],
        "Số lượng khách": summary["customers"],
        "Tỷ lệ khách (%)": (summary["customer_share"] * 100).round(2).astype(str) + "%",
        "Recency trung vị (ngày)": summary["recency_median"].round(1),
        "Frequency trung vị (đơn)": summary["frequency_median"].round(1),
        "Monetary trung vị (£)": summary["monetary_median"].map("{:,.2f} £".format),
        "Tổng doanh thu (£)": summary["monetary_total"].map("{:,.2f} £".format),
        "Tỷ trọng doanh thu (%)": (summary["revenue_share"] * 100).round(2).astype(str) + "%",
    })
    profile_vi_path = output_table_dir / "ho_so_chan_dung_khach_hang.csv"
    summary_display.to_csv(profile_vi_path, index=False, encoding="utf-8-sig")
    print(f" -> Đã lưu bảng hồ sơ phân khúc tiếng Việt: {profile_vi_path}")
    print("\n--- BẢNG HỒ SƠ PHÂN KHÚC KHÁCH HÀNG (RFM GỐC) ---")
    print(summary_display.to_string(index=False))

    # 1. Trực quan hóa bản đồ phân cụm trên không gian PCA 2 chiều
    X = features_df[FEATURE_COLUMNS].to_numpy()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=df,
        x="PCA1",
        y="PCA2",
        hue="persona",
        palette="tab10",
        alpha=0.75,
        s=45,
        edgecolor=None,
    )
    plt.title(
        f"Bản Đồ Phân Cụm Khách Hàng (PCA 2D - Giải thích {pca.explained_variance_ratio_.sum()*100:.1f}% Phương Sai)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel(f"Thành phần chính 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Thành phần chính 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    pca_path = output_fig_dir / "pca_clusters.png"
    plt.savefig(pca_path)
    plt.close()
    print(f" -> Đã lưu bản đồ phân cụm PCA 2D: {pca_path}")

    # 2. Phân bố đặc trưng RFM theo cụm (Biểu đồ hộp Boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=df, x="cluster", y="Recency", hue="cluster", ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("Recency (Số ngày kể từ lần mua gần nhất - Càng thấp càng tốt)")
    axes[0].set_xlabel("Mã số cụm (Cluster ID)")
    axes[0].set_ylabel("Số ngày")

    sns.boxplot(data=df, x="cluster", y="Frequency", hue="cluster", ax=axes[1], palette="Set2", legend=False)
    axes[1].set_yscale("log")
    axes[1].set_title("Frequency (Số lượng hóa đơn - Thang đo Logarit)")
    axes[1].set_xlabel("Mã số cụm (Cluster ID)")
    axes[1].set_ylabel("Số đơn hàng")

    sns.boxplot(data=df, x="cluster", y="Monetary", hue="cluster", ax=axes[2], palette="Set2", legend=False)
    axes[2].set_yscale("log")
    axes[2].set_title("Monetary (Tổng giá trị chi tiêu £ - Thang đo Logarit)")
    axes[2].set_xlabel("Mã số cụm (Cluster ID)")
    axes[2].set_ylabel("Số tiền chi tiêu (£)")

    plt.suptitle("Phân Phối Đặc Trưng RFM Theo Từng Cụm Khách Hàng", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    rfm_dist_path = output_fig_dir / "rfm_distribution.png"
    plt.savefig(rfm_dist_path)
    plt.close()
    print(f" -> Đã lưu biểu đồ phân phối RFM Boxplots: {rfm_dist_path}")

    # 3. Biểu đồ cột so sánh trung vị Recency, Frequency, Monetary giữa các cụm
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.barplot(data=summary, x="cluster", y="recency_median", hue="cluster", ax=axes[0], palette="Blues_r", legend=False)
    axes[0].set_title("Recency trung vị (Số ngày)")
    axes[0].set_xlabel("Mã số cụm")
    axes[0].set_ylabel("Số ngày")

    sns.barplot(data=summary, x="cluster", y="frequency_median", hue="cluster", ax=axes[1], palette="Greens", legend=False)
    axes[1].set_title("Frequency trung vị (Số đơn hàng)")
    axes[1].set_xlabel("Mã số cụm")
    axes[1].set_ylabel("Số đơn hàng")

    sns.barplot(data=summary, x="cluster", y="monetary_median", hue="cluster", ax=axes[2], palette="Oranges", legend=False)
    axes[2].set_title("Monetary trung vị (Tổng chi tiêu £)")
    axes[2].set_xlabel("Mã số cụm")
    axes[2].set_ylabel("Tổng chi tiêu (£)")

    plt.suptitle("So Sánh Giá Trị Trung Vị RFM Giữa Các Cụm Khách Hàng", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    bar_path = output_fig_dir / "cluster_profiles.png"
    plt.savefig(bar_path)
    plt.close()
    print(f" -> Đã lưu biểu đồ so sánh trung vị RFM: {bar_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4, help="Số lượng cụm tối ưu cần phân chia (mặc định: 4)")
    parser.add_argument("--linkage", type=str, default="ward", choices=["ward", "complete", "average"], help="Tiêu chí liên kết phân cấp (mặc định: ward)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_cleaned", help="Thư mục chứa dữ liệu đã làm sạch")
    args = parser.parse_args()

    setup_plotting_style()
    dirs = create_directories()

    print("=" * 70)
    print("QUY TRÌNH PHÂN CỤM PHÂN CẤP TÍCH LŨY (AGGLOMERATIVE CLUSTERING)")
    print("Phân khúc Khách hàng Bán lẻ Trực tuyến (Online Retail Customer Segmentation)")
    print(f"Cấu hình đã chọn: Số cụm k={args.k}, Tiêu chí liên kết linkage='{args.linkage}'")
    print("=" * 70)

    # 1. Nạp dữ liệu giai đoạn Development
    rfm_dev, feat_dev, outcomes_dev = load_phase(args.data_dir, "development")
    X_dev = feat_dev[FEATURE_COLUMNS].to_numpy()
    print(f"Đã nạp dữ liệu Development: {len(rfm_dev):,} khách hàng, các đặc trưng: {FEATURE_COLUMNS}")

    # 2. Xây dựng Dendrogram và phân tích cây phân cấp
    plot_dendrograms(X_dev, dirs["figures"])

    # 3. Tìm kiếm lưới (Grid Search) qua các liên kết và số cụm k (2..10)
    grid_df = grid_search_hyperparameters(
        X_dev,
        k_range=range(2, 11),
        linkage_methods=["ward", "complete", "average"],
        output_table_dir=dirs["tables"],
        output_fig_dir=dirs["figures"],
    )

    # 4. Huấn luyện mô hình tối ưu cho giai đoạn Development
    print("\n" + "=" * 70)
    print(f"BƯỚC 4: Huấn luyện mô hình & Xuất kết quả cho giai đoạn 'Development'")
    print("=" * 70)
    labels_dev, meta_dev = fit_and_export_model(
        phase="development",
        features_df=feat_dev,
        rfm_df=rfm_dev,
        best_k=args.k,
        best_linkage=args.linkage,
        output_dir=dirs["dev_out"],
    )

    # 5. Phác họa chân dung khách hàng (Development)
    generate_customer_profiles(
        rfm_df=rfm_dev,
        features_df=feat_dev,
        labels=labels_dev,
        output_fig_dir=dirs["figures"],
        output_table_dir=dirs["tables"],
    )

    # 6. Huấn luyện mô hình tối ưu cho giai đoạn Final
    print("\n" + "=" * 70)
    print(f"BƯỚC 5: Huấn luyện mô hình & Xuất kết quả cho giai đoạn 'Final'")
    print("=" * 70)
    rfm_final, feat_final, outcomes_final = load_phase(args.data_dir, "final")
    print(f"Đã nạp dữ liệu Final: {len(rfm_final):,} khách hàng")
    labels_final, meta_final = fit_and_export_model(
        phase="final",
        features_df=feat_final,
        rfm_df=rfm_final,
        best_k=args.k,
        best_linkage=args.linkage,
        output_dir=dirs["final_out"],
    )

    # 7. Đánh giá ngoài thời gian (Out-of-time 90 ngày tương lai)
    print("\n" + "=" * 70)
    print("BƯỚC 6: Kiểm định ngoài thời gian (Hành vi mua sắm trong 90 ngày tiếp theo)")
    print("=" * 70)
    from retail_segmentation.temporal_evaluation import evaluate_submitted_models
    temporal_dev_dir = ROOT / "reports" / "temporal" / "development"
    scores, profiles = evaluate_submitted_models(
        args.data_dir, dirs["dev_out"], temporal_dev_dir, "development"
    )

    if not profiles.empty:
        persona_map = {
            "2": "Khách hàng VIP / Quán quân (Chi tiêu khủng)",
            "0": "Khách hàng trung thành (Gắn kết đều)",
            "3": "Khách hàng tiềm năng (Mới mua gần đây)",
            "1": "Khách hàng có nguy cơ rời bỏ / Ngủ đông",
        }
        profiles_vi = pd.DataFrame({
            "Mã cụm": profiles["cluster"].astype(str),
            "Chân dung khách hàng": profiles["cluster"].astype(str).map(persona_map).fillna(profiles["cluster"]),
            "Số lượng khách": profiles["customers"],
            "Số người mua lại": profiles["repurchased"],
            "Tỷ lệ mua lại (%)": (profiles["repurchase_rate"] * 100).round(2).astype(str) + "%",
            "Doanh thu 90 ngày (£)": profiles["future_revenue"].map("{:,.2f} £".format),
            "Doanh thu TB/Khách (£)": profiles["future_revenue_per_customer"].map("{:,.2f} £".format),
            "Tỷ trọng doanh thu tương lai (%)": (profiles["future_revenue_share"] * 100).round(2).astype(str) + "%",
        })
        print("--- KẾT QUẢ THEO DÕI HÀNH VI 90 NGÀY TƯƠNG LAI (GIAI ĐOẠN DEVELOPMENT) ---")
        print(profiles_vi.to_string(index=False))

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TOÀN BỘ QUY TRÌNH PHÂN CỤM AGGLOMERATIVE VÀ KIỂM ĐỊNH THÀNH CÔNG!")
    print("=" * 70)


if __name__ == "__main__":
    main()
