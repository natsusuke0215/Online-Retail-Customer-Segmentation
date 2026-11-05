"""
Xây dựng và đánh giá mô hình phân cụm DBSCAN cho bài toán Phân khúc Khách hàng Online Retail.

Nhiệm vụ:
1. Tối ưu hai tham số eps và min_samples (k-distance Knee/Elbow + Grid Search trên tập Development).
2. Phân tích các cụm khách hàng và bóc tách chuyên sâu những khách hàng bị xem là nhiễu (Noise = -1).
3. Đánh giá khả năng phát hiện cụm có hình dạng khác nhau (trên không gian RFM thực tế và bộ dữ liệu kiểm thử hình học).
4. Xuất nhãn cụm chuẩn cho hai phase Development và Final theo hợp đồng dữ liệu của dự án.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def resolve_repo_root() -> Path:
    """Tự động xác định thư mục gốc của repository Online-Retail-Customer-Segmentation."""
    here = Path(__file__).resolve().parent
    if (here / "data_cleaned").exists() and (here / "src" / "retail_segmentation").exists():
        return here
    candidate = here / "Online-Retail-Customer-Segmentation"
    if (candidate / "data_cleaned").exists():
        return candidate
    raise FileNotFoundError("Không tìm thấy thư mục Online-Retail-Customer-Segmentation/data_cleaned.")


ROOT = resolve_repo_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retail_segmentation.evaluation import evaluate_labels  # noqa: E402
from retail_segmentation.pipeline import FEATURE_COLUMNS, RFM_COLUMNS  # noqa: E402
from retail_segmentation.temporal_evaluation import (  # noqa: E402
    evaluate_cluster_priority,
    evaluate_simple_rules,
    load_phase,
)

FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"
PALETTE = ["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
NOISE_COLOR = "#dc2626"


def find_kneedle_elbow(curve: np.ndarray, trim_top_fraction: float = 0.0) -> tuple[int, float]:
    """Tìm điểm gãy (Knee/Elbow) trên đường cong k-distance đã sắp xếp tăng dần bằng khoảng cách vuông góc cực đại."""
    n = len(curve)
    limit = int(round(n * (1.0 - trim_top_fraction))) if trim_top_fraction > 0 else n
    y = curve[: max(2, limit)]
    x_norm = np.linspace(0.0, 1.0, len(y))
    y_min, y_max = float(y.min()), float(y.max())
    if y_max <= y_min:
        return 0, float(y[0])
    y_norm = (y - y_min) / (y_max - y_min)
    # Đường thẳng nối (0, 0) và (1, 1) có phương trình y = x; khoảng cách thẳng đứng là x_norm - y_norm
    distance_to_chord = x_norm - y_norm
    knee_idx = int(np.argmax(distance_to_chord))
    return knee_idx, float(y[knee_idx])


def compute_k_distance_summary(
    X: np.ndarray, min_samples_list: list[int]
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Tính khoảng cách tới láng giềng thứ k và điểm gãy Kneedle cho từng giá trị min_samples."""
    records = []
    curves: dict[int, np.ndarray] = {}
    for ms in min_samples_list:
        nn = NearestNeighbors(n_neighbors=ms, metric="euclidean")
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        k_dist = np.sort(dists[:, -1])
        curves[ms] = k_dist
        knee_idx_all, knee_eps_all = find_kneedle_elbow(k_dist, trim_top_fraction=0.0)
        knee_idx_trim, knee_eps_trim = find_kneedle_elbow(k_dist, trim_top_fraction=0.005)
        pcts = np.percentile(k_dist, [50, 75, 85, 90, 95, 97, 99])
        records.append({
            "min_samples": ms,
            "p50": round(float(pcts[0]), 4),
            "p75": round(float(pcts[1]), 4),
            "p85": round(float(pcts[2]), 4),
            "p90": round(float(pcts[3]), 4),
            "p95": round(float(pcts[4]), 4),
            "p97": round(float(pcts[5]), 4),
            "p99": round(float(pcts[6]), 4),
            "knee_index_full": knee_idx_all,
            "knee_quantile_full": round(knee_idx_all / len(X), 4),
            "knee_eps_full": round(knee_eps_all, 4),
            "knee_index_trim995": knee_idx_trim,
            "knee_quantile_trim995": round(knee_idx_trim / len(X), 4),
            "knee_eps_trim995": round(knee_eps_trim, 4),
        })
    return pd.DataFrame(records), curves


def run_hyperparameter_grid(
    X: np.ndarray,
    rfm: pd.DataFrame,
    outcomes: pd.DataFrame,
    eps_values: list[float],
    min_samples_values: list[int],
) -> pd.DataFrame:
    """Quét lưới (eps, min_samples) trên tập Development và thu thập các thước đo hình học + hành vi."""
    rows = []
    customer_ids = rfm["CustomerID"]
    for ms in min_samples_values:
        for eps in eps_values:
            t0 = time.perf_counter()
            labels = DBSCAN(eps=eps, min_samples=ms, metric="euclidean").fit_predict(X)
            elapsed = time.perf_counter() - t0
            ev = evaluate_labels(X, labels, model_name="dbscan", runtime_seconds=elapsed)
            sizes = sorted(ev["cluster_sizes"].values(), reverse=True)
            largest_share = (sizes[0] / ev["assigned_customers"]) if ev["assigned_customers"] > 0 and sizes else np.nan
            small_clusters = sum(1 for c_size in sizes if c_size < 15)
            macro_clusters = sum(1 for c_size in sizes if c_size >= 15)

            lbl_df = pd.DataFrame({"CustomerID": customer_ids, "cluster": labels.astype(str)})
            pol = evaluate_cluster_priority(rfm, outcomes, lbl_df, "dbscan")

            # Tính điểm tổng hợp chọn tham số nội tại (chỉ dựa trên hình học và độ sạch của cụm trên Development):
            # Ưu tiên: 3-4 cụm lớn (không có cụm vụn < 15 khách), tỷ lệ nhiễu trong khoảng [3%, 10%], Silhouette & CHI cao, DBI thấp.
            valid_regime = (
                2 <= ev["n_clusters"] <= 6
                and 0.025 <= ev["noise_fraction"] <= 0.12
                and largest_share <= 0.60
                and small_clusters == 0
                and ev["n_clusters"] >= 3
            )
            rows.append({
                "eps": round(float(eps), 2),
                "min_samples": int(ms),
                "n_clusters": ev["n_clusters"],
                "macro_clusters_ge15": macro_clusters,
                "splinter_clusters_lt15": small_clusters,
                "noise_customers": ev["noise_customers"],
                "noise_fraction": round(ev["noise_fraction"], 4),
                "silhouette": round(float(ev["silhouette"]), 4) if np.isfinite(ev["silhouette"]) else np.nan,
                "davies_bouldin": round(float(ev["davies_bouldin"]), 4) if np.isfinite(ev["davies_bouldin"]) else np.nan,
                "calinski_harabasz": round(float(ev["calinski_harabasz"]), 2) if np.isfinite(ev["calinski_harabasz"]) else np.nan,
                "largest_cluster_share": round(float(largest_share), 4) if np.isfinite(largest_share) else np.nan,
                "clean_macro_regime": bool(valid_regime),
                "dev_selected_repurchase_rate_15pct": round(float(pol["selected_repurchase_rate"]), 4),
                "dev_future_revenue_share_15pct": round(float(pol["future_revenue_share"]), 4),
                "runtime_seconds": round(float(elapsed), 5),
                "cluster_sizes": json.dumps(ev["cluster_sizes"], ensure_ascii=False),
            })
    return pd.DataFrame(rows)


def analyze_clusters_and_noise(
    rfm: pd.DataFrame,
    feat: pd.DataFrame,
    outcomes: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tạo bảng hồ sơ chi tiết cho từng cụm và phân rã chuyên sâu nhóm khách hàng Nhiễu (cluster = -1)."""
    df = rfm.merge(feat, on="CustomerID", validate="one_to_one").merge(
        outcomes, on="CustomerID", validate="one_to_one"
    )
    df["cluster"] = labels.astype(int)

    total_customers = len(df)
    total_past_m = float(df["Monetary"].sum())
    total_future_m = float(df["future_revenue"].sum())

    cluster_names = {
        -1: "Noise (-1): Ngoại lệ (Siêu VIP & Đơn lẻ bất thường)",
        0: "Cụm 0: Khách hàng Trung thành Cốt lõi (F >= 3)",
        1: "Cụm 1: Khách hàng Mua 1 lần / Ngủ đông (F = 1)",
        2: "Cụm 2: Khách hàng Lặp lại Bước đầu (F = 2)",
        3: "Cụm 3: Nhóm Hoạt động Cao Mới nhất (R = 1, F cao)",
    }

    profile_rows = []
    for cid, grp in df.groupby("cluster", sort=True):
        profile_rows.append({
            "cluster": int(cid),
            "cluster_label": cluster_names.get(int(cid), f"Cụm {cid}"),
            "customers": int(len(grp)),
            "customer_share": round(len(grp) / total_customers, 4),
            "recency_median": round(float(grp["Recency"].median()), 2),
            "recency_iqr": round(float(grp["Recency"].quantile(0.75) - grp["Recency"].quantile(0.25)), 2),
            "frequency_median": round(float(grp["Frequency"].median()), 2),
            "frequency_min": int(grp["Frequency"].min()),
            "frequency_max": int(grp["Frequency"].max()),
            "monetary_median": round(float(grp["Monetary"].median()), 2),
            "monetary_mean": round(float(grp["Monetary"].mean()), 2),
            "monetary_total": round(float(grp["Monetary"].sum()), 2),
            "past_monetary_share": round(float(grp["Monetary"].sum()) / total_past_m, 4),
            "repurchase_rate_90d": round(float(grp["repurchased_in_90d"].mean()), 4),
            "future_revenue_mean": round(float(grp["future_revenue"].mean()), 2),
            "future_revenue_total": round(float(grp["future_revenue"].sum()), 2),
            "future_revenue_share": round(float(grp["future_revenue"].sum()) / total_future_m, 4),
        })
    cluster_profile_df = pd.DataFrame(profile_rows)

    # Phân rã nhóm Noise (-1) thành 4 nhóm con có ý nghĩa kinh doanh rõ rệt
    q90_m = float(df["Monetary"].quantile(0.90))
    q25_m = float(df["Monetary"].quantile(0.25))
    noise_df = df.loc[df["cluster"] == -1].copy()
    noise_df["noise_subtype"] = np.select(
        [
            (noise_df["Monetary"] >= q90_m) & (noise_df["Frequency"] >= 5),
            (noise_df["Monetary"] >= q90_m) & (noise_df["Frequency"] < 5),
            (noise_df["Monetary"] < q25_m),
        ],
        [
            "1. Siêu VIP / Khách sỉ Cá voi (F >= 5 & Monetary >= P90)",
            "2. Khách mua Lô lớn Ít lần (F < 5 & Monetary >= P90)",
            "3. Khách chi tiêu Siêu nhỏ (Monetary < P25)",
        ],
        default="4. Tổ hợp RFM Lệch biên (Mua rất gần R=1 hoặc lệch tỷ lệ F/M)",
    )

    noise_summary_rows = []
    for stype, grp in noise_df.groupby("noise_subtype", sort=True):
        noise_summary_rows.append({
            "noise_subtype": stype,
            "customers": int(len(grp)),
            "share_of_noise": round(len(grp) / max(1, len(noise_df)), 4),
            "share_of_all_customers": round(len(grp) / total_customers, 4),
            "recency_median": round(float(grp["Recency"].median()), 2),
            "frequency_median": round(float(grp["Frequency"].median()), 2),
            "monetary_median": round(float(grp["Monetary"].median()), 2),
            "monetary_mean": round(float(grp["Monetary"].mean()), 2),
            "monetary_total": round(float(grp["Monetary"].sum()), 2),
            "share_of_all_past_monetary": round(float(grp["Monetary"].sum()) / total_past_m, 4),
            "repurchase_rate_90d": round(float(grp["repurchased_in_90d"].mean()), 4),
            "future_revenue_total": round(float(grp["future_revenue"].sum()), 2),
            "share_of_all_future_revenue": round(float(grp["future_revenue"].sum()) / total_future_m, 4),
        })
    noise_summary_df = pd.DataFrame(noise_summary_rows)
    return df, cluster_profile_df, noise_summary_df


def evaluate_cluster_geometry_and_shapes(
    X: np.ndarray, dbscan_labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Đánh giá hình học cụm trên không gian RFM 3D (Eigenvalue ratios, Elongation, Planarity)
    và thực nghiệm đối chứng khả năng phát hiện cụm có hình dạng khác nhau giữa DBSCAN và K-Means.
    """
    geom_rows = []
    for cid in sorted(np.unique(dbscan_labels)):
        pts = X[dbscan_labels == cid]
        if len(pts) < 3:
            continue
        cov = np.cov(pts, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(eigvals, 0.0)
        l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
        elongation = 1.0 - np.sqrt(l2 / l1) if l1 > 0 else 0.0
        planarity = 1.0 - np.sqrt(l3 / l2) if l2 > 1e-12 else 1.0
        sphericity = np.sqrt(l3 / l1) if l1 > 0 else 0.0
        if cid == -1:
            shape_desc = "Phân tán bao quanh rìa không gian 3D (Outlier Halo)"
        elif planarity > 0.99:
            shape_desc = "Tấm phẳng / Dải băng 2D suy biến chiều F (2D Planar Ribbon)"
        elif elongation > 0.35 or sphericity < 0.45:
            shape_desc = "Ống / Đám mây thuôn dài 3D theo trục F-M (Elongated 3D Manifold)"
        else:
            shape_desc = "Đám mây 3D bán cầu (Compact 3D Cloud)"

        geom_rows.append({
            "cluster": int(cid),
            "customers": int(len(pts)),
            "eigval_1": round(l1, 4),
            "eigval_2": round(l2, 4),
            "eigval_3": round(l3, 6),
            "elongation_index": round(float(elongation), 4),
            "planarity_index": round(float(planarity), 4),
            "sphericity_index": round(float(sphericity), 4),
            "shape_characterization": shape_desc,
        })
    rfm_geom_df = pd.DataFrame(geom_rows)

    # Đối sánh trực tiếp với K-Means (k=3) trên dữ liệu RFM Development
    km_rfm = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(X)
    non_noise_mask = dbscan_labels != -1
    ari_rfm_vs_kmeans = adjusted_rand_score(dbscan_labels[non_noise_mask], km_rfm[non_noise_mask])

    # Thực nghiệm trên 4 cấu trúc hình học chuẩn để định lượng khả năng phát hiện hình dạng
    rng = np.random.default_rng(42)

    # 1. Two Moons (Phi lồi - Non-convex)
    X_moons, y_moons = make_moons(n_samples=600, noise=0.07, random_state=42)
    X_moons = StandardScaler().fit_transform(X_moons)

    # 2. Concentric Circles (Vòng đồng tâm lồng nhau)
    X_circles, y_circles = make_circles(n_samples=600, factor=0.45, noise=0.05, random_state=42)
    X_circles = StandardScaler().fit_transform(X_circles)

    # 3. Anisotropic Elongated Streaks + Noise (Dải băng song song kéo dài tương tự mặt phẳng Frequency của RFM)
    t_line = rng.uniform(-2.4, 2.4, size=(600,))
    ribbon_id = np.repeat([0, 1, 2], 200)
    offsets = np.array([-1.25, 0.0, 1.25])[ribbon_id]
    X_aniso_core = np.column_stack([
        t_line + rng.normal(0, 0.11, size=600),
        0.85 * t_line + offsets + rng.normal(0, 0.11, size=600),
    ])
    noise_pts = rng.uniform(low=-3.6, high=3.6, size=(40, 2))
    X_aniso = StandardScaler().fit_transform(np.vstack([X_aniso_core, noise_pts]))
    y_aniso = np.concatenate([ribbon_id, np.full(40, -1)])

    # 4. Varying Density Clusters (Mật độ biến thiên mạnh - minh họa giới hạn của DBSCAN với 1 eps toàn cục)
    X_var, y_var = make_blobs(
        n_samples=[280, 220, 100],
        centers=[[-1.35, -0.85], [-0.55, -0.15], [1.85, 1.35]],
        cluster_std=[0.14, 0.16, 0.85],
        random_state=42,
    )
    X_var = StandardScaler().fit_transform(X_var)

    benchmarks = [
        ("1. Two Moons (Phi lồi / Non-convex)", X_moons, y_moons, 0.32, 5, 2),
        ("2. Concentric Circles (Vòng đồng tâm)", X_circles, y_circles, 0.30, 5, 2),
        ("3. Anisotropic Ribbons + Noise (Dải băng thon dài)", X_aniso, y_aniso, 0.22, 6, 3),
        ("4. Varying Density (Mật độ chênh lệch mạnh)", X_var, y_var, 0.20, 8, 3),
    ]

    bench_rows = []
    bench_plot_data = {}
    for name, X_b, y_true, eps_b, ms_b, k_b in benchmarks:
        y_db = DBSCAN(eps=eps_b, min_samples=ms_b).fit_predict(X_b)
        y_km = KMeans(n_clusters=k_b, random_state=42, n_init=20).fit_predict(X_b)
        valid_true = y_true != -1
        # Khi tính ARI/NMI trên các điểm thuộc cụm thật, các điểm bị DBSCAN đánh dấu là nhiễu (-1)
        # không tạo thành một cụm thực sự mà là các điểm đơn lẻ chưa được phân cụm (singletons)
        y_db_eval = y_db[valid_true].copy()
        noise_positions = np.flatnonzero(y_db_eval == -1)
        if len(noise_positions) > 0:
            y_db_eval[noise_positions] = -np.arange(1, len(noise_positions) + 1)
        ari_db = adjusted_rand_score(y_true[valid_true], y_db_eval)
        nmi_db = normalized_mutual_info_score(y_true[valid_true], y_db_eval)
        ari_km = adjusted_rand_score(y_true[valid_true], y_km[valid_true])
        nmi_km = normalized_mutual_info_score(y_true[valid_true], y_km[valid_true])
        bench_rows.append({
            "geometry_scenario": name,
            "dbscan_params": f"eps={eps_b}, min_samples={ms_b}",
            "dbscan_clusters_found": int(len(set(y_db) - {-1})),
            "dbscan_noise_points": int((y_db == -1).sum()),
            "dbscan_ARI": round(float(ari_db), 4),
            "dbscan_NMI": round(float(nmi_db), 4),
            "kmeans_k": k_b,
            "kmeans_ARI": round(float(ari_km), 4),
            "kmeans_NMI": round(float(nmi_km), 4),
            "ari_advantage_dbscan_minus_kmeans": round(float(ari_db - ari_km), 4),
        })
        bench_plot_data[name] = (X_b, y_true, y_db, y_km)

    bench_df = pd.DataFrame(bench_rows)
    extra_info = {
        "km_rfm_labels": km_rfm,
        "ari_rfm_vs_kmeans": round(float(ari_rfm_vs_kmeans), 4),
        "bench_plot_data": bench_plot_data,
    }
    return rfm_geom_df, bench_df, extra_info


def generate_all_figures(
    dev_df: pd.DataFrame,
    X_dev: np.ndarray,
    k_dist_df: pd.DataFrame,
    k_curves: dict[int, np.ndarray],
    grid_df: pd.DataFrame,
    cluster_profile_df: pd.DataFrame,
    noise_summary_df: pd.DataFrame,
    shape_extra: dict,
    chosen_eps: float,
    chosen_ms: int,
) -> list[Path]:
    """Tạo bộ 6 biểu đồ phân tích trực quan chất lượng cao phục vụ báo cáo và notebook."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved_paths: list[Path] = []

    # 1. Hình 1: Phân phối RFM sau chuẩn hóa và cấu trúc mặt phẳng rời rạc của Frequency
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, col, color in zip(axes[:2], ["Recency_scaled", "Monetary_scaled"], [PALETTE[0], PALETTE[1]]):
        sns.histplot(dev_df[col], bins=35, kde=True, color=color, ax=ax, edgecolor="white")
        ax.set_title(f"Phân phối {col} (Development)", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Số khách hàng")
    sc = axes[2].scatter(
        dev_df["Frequency_scaled"],
        dev_df["Monetary_scaled"],
        c=dev_df["Recency_scaled"],
        cmap="viridis",
        s=14,
        alpha=0.65,
    )
    for f_val, label_txt, y_txt in [(1, "F=1", 4.35), (2, "F=2", 3.75), (3, "F=3", 4.35), (4, "F=4", 3.75)]:
        f_sc = float(dev_df.loc[dev_df["Frequency"] == f_val, "Frequency_scaled"].iloc[0])
        axes[2].axvline(f_sc, color="#ef4444", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[2].text(f_sc + 0.04, y_txt, label_txt, color="#b91c1c", fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Recency_scaled")
    axes[2].set_title("Cấu trúc dải rời rạc Frequency vs Monetary", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Frequency_scaled (các vạch đỏ: F=1, 2, 3, 4)")
    axes[2].set_ylabel("Monetary_scaled")
    fig.suptitle("Hình 1: Đặc trưng RFM chuẩn hóa và tính chất lượng tử hóa của Frequency (n = 2,716)", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    p1 = FIGURES_DIR / "dbscan_01_rfm_feature_space.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p1)

    # 2. Hình 2: Đồ thị k-distance và điểm gãy Kneedle (Elbow)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ms, color in zip([4, 6, 8, 10, 12, 15], PALETTE):
        curve = k_curves[ms]
        q_axis = np.linspace(0, 100, len(curve))
        axes[0].plot(q_axis, curve, label=f"k = {ms}", linewidth=1.8, color=color)
    axes[0].axhline(chosen_eps, color=NOISE_COLOR, linestyle="--", linewidth=1.5, label=f"eps chọn = {chosen_eps:.2f}")
    axes[0].set_xlim(50, 100)
    axes[0].set_ylim(0, 1.35)
    axes[0].set_title("Đường cong k-distance (Phân vị 50% - 100%)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Phân vị khách hàng (%)")
    axes[0].set_ylabel("Khoảng cách Euclid tới láng giềng thứ k")
    axes[0].legend(loc="upper left", frameon=True)

    # Chi tiết đường cong tại min_samples = chosen_ms
    curve_sel = k_curves[chosen_ms]
    idx_knee, eps_knee = find_kneedle_elbow(curve_sel, trim_top_fraction=0.0)
    axes[1].plot(np.arange(len(curve_sel)), curve_sel, color=PALETTE[0], linewidth=2.2, label=f"{chosen_ms}-distance curve")
    axes[1].scatter([idx_knee], [eps_knee], color=NOISE_COLOR, s=90, zorder=5, label=f"Điểm gãy Kneedle: idx={idx_knee} (eps={eps_knee:.3f})")
    axes[1].axhline(chosen_eps, color="#0f766e", linestyle="-.", linewidth=1.6, label=f"Ngưỡng chốt eps = {chosen_eps:.2f}")
    axes[1].axvline(idx_knee, color=NOISE_COLOR, linestyle=":", linewidth=1.2)
    axes[1].annotate(
        f"Điểm gãy (Knee)\neps ≈ {eps_knee:.3f} → chọn {chosen_eps:.2f}\n({100 - idx_knee/len(curve_sel)*100:.1f}% vùng thưa)",
        xy=(idx_knee, eps_knee),
        xytext=(idx_knee - 750, eps_knee + 0.42),
        arrowprops={"arrowstyle": "->", "color": NOISE_COLOR, "lw": 1.5},
        fontsize=9.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fef2f2", "edgecolor": NOISE_COLOR},
    )
    axes[1].set_title(f"Xác định điểm gãy Kneedle tại min_samples = {chosen_ms}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Thứ tự khách hàng (sắp xếp theo khoảng cách tăng dần)")
    axes[1].set_ylabel(f"Khoảng cách tới láng giềng thứ {chosen_ms}")
    axes[1].legend(loc="upper left", frameon=True)
    fig.suptitle("Hình 2: Tối ưu tham số eps bằng phương pháp đồ thị k-distance (Knee/Elbow)", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    p2 = FIGURES_DIR / "dbscan_02_k_distance_elbow.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Hình 3: Heatmap kết quả Grid Search trên lưới (eps, min_samples)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics_cfg = [
        ("n_clusters", "Số lượng cụm tìm được (k)", "YlGnBu", ".0f"),
        ("noise_fraction", "Tỷ lệ khách hàng nhiễu (Noise Fraction)", "OrRd", ".2f"),
        ("silhouette", "Chỉ số Silhouette (không tính nhiễu)", "viridis", ".3f"),
        ("calinski_harabasz", "Chỉ số Calinski-Harabasz (CHI)", "magma", ".0f"),
    ]
    for ax, (col, title, cmap, fmt) in zip(axes.flat, metrics_cfg):
        piv = grid_df.pivot(index="min_samples", columns="eps", values=col)
        sns.heatmap(piv, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar_kws={"shrink": 0.82}, annot_kws={"size": 8})
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Bán kính lân cận (eps)")
        ax.set_ylabel("Số điểm tối thiểu (min_samples)")
    fig.suptitle("Hình 3: Tìm kiếm lưới (Grid Search) hai siêu tham số eps và min_samples trên tập Development", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    p3 = FIGURES_DIR / "dbscan_03_grid_search_heatmaps.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Hình 4: Biểu đồ phân tán 2D PCA (theo nhãn cụm và theo biến rời rạc) + 3D RFM
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dev)
    var_r = pca.explained_variance_ratio_ * 100

    fig = plt.figure(figsize=(16, 5.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    cluster_colors = {-1: NOISE_COLOR, 0: PALETTE[0], 1: PALETTE[1], 2: PALETTE[2], 3: PALETTE[3]}
    cluster_short = {
        -1: "Noise (-1): Ngoại lệ/VIP",
        0: "Cụm 0: Trung thành (F>=3)",
        1: "Cụm 1: Mua 1 lần (F=1)",
        2: "Cụm 2: Mua 2 lần (F=2)",
    }
    for cid in sorted(dev_df["cluster"].unique()):
        sub = dev_df["cluster"] == cid
        ax1.scatter(
            X_pca[sub, 0],
            X_pca[sub, 1],
            s=22 if cid == -1 else 12,
            alpha=0.85 if cid == -1 else 0.55,
            color=cluster_colors.get(int(cid), "#64748b"),
            marker="x" if cid == -1 else "o",
            label=f"{cluster_short.get(int(cid), f'Cụm {cid}')} (n={int(sub.sum())})",
        )
    ax1.set_title(f"PCA 2D theo Nhãn cụm DBSCAN\n(eps={chosen_eps}, min_samples={chosen_ms})", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel(f"PC1 ({var_r[0]:.1f}% phương sai)")
    ax1.set_ylabel(f"PC2 ({var_r[1]:.1f}% phương sai)")
    ax1.legend(loc="upper right", fontsize=8, frameon=True)

    # Scatterplot trên PCA 2D với biến rời rạc (Trạng thái mua lại 90 ngày & Mức tần suất) làm hue
    rep_labels = np.where(dev_df["repurchased_in_90d"] == 1, "Có mua lại trong 90 ngày (1)", "Không mua lại (0)")
    for val, color, label in [(0, "#94a3b8", "Không mua lại 90 ngày (0)"), (1, "#059669", "Có mua lại 90 ngày (1)")]:
        sub = dev_df["repurchased_in_90d"] == val
        ax2.scatter(
            X_pca[sub, 0],
            X_pca[sub, 1],
            s=12,
            alpha=0.55,
            color=color,
            label=f"{label} (n={int(sub.sum())})",
        )
    ax2.set_title("PCA 2D theo Biến rời rạc (Mua lại trong 90 ngày)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel(f"PC1 ({var_r[0]:.1f}% phương sai)")
    ax2.set_ylabel(f"PC2 ({var_r[1]:.1f}% phương sai)")
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    for cid in sorted(dev_df["cluster"].unique()):
        sub = dev_df["cluster"] == cid
        ax3.scatter(
            dev_df.loc[sub, "Recency_scaled"],
            dev_df.loc[sub, "Frequency_scaled"],
            dev_df.loc[sub, "Monetary_scaled"],
            s=18 if cid == -1 else 8,
            alpha=0.85 if cid == -1 else 0.45,
            color=cluster_colors.get(int(cid), "#64748b"),
            marker="^" if cid == -1 else "o",
        )
    ax3.set_title("Không gian 3D (R, F, M) chuẩn hóa", fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("Recency_scaled")
    ax3.set_ylabel("Frequency_scaled")
    ax3.set_zlabel("Monetary_scaled")
    ax3.view_init(elev=22, azim=-52)

    fig.suptitle("Hình 4: Trực quan hóa kết quả phân cụm DBSCAN trên không gian PCA 2D và RFM 3D", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    p4 = FIGURES_DIR / "dbscan_04_pca_and_3d_clusters.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    # 5. Hình 5: Phân tích hồ sơ các cụm và bóc tách chuyên sâu nhóm Nhiễu (Noise = -1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    order_c = [1, 2, 0, -1]
    labels_x = ["Cụm 1\n(F=1)", "Cụm 2\n(F=2)", "Cụm 0\n(F>=3)", "Noise (-1)\n(Ngoại lệ)"]
    prof_indexed = cluster_profile_df.set_index("cluster").loc[order_c]

    x_pos = np.arange(len(order_c))
    w = 0.26
    axes[0].bar(x_pos - w, prof_indexed["customer_share"] * 100, width=w, color=PALETTE[0], label="% Số khách hàng")
    axes[0].bar(x_pos, prof_indexed["past_monetary_share"] * 100, width=w, color=PALETTE[1], label="% Doanh thu quá khứ")
    axes[0].bar(x_pos + w, prof_indexed["future_revenue_share"] * 100, width=w, color=PALETTE[2], label="% Doanh thu 90 ngày tới")
    axes[0].set_xticks(x_pos, labels_x)
    axes[0].set_ylabel("Tỷ trọng (%)")
    axes[0].set_title("Tỷ trọng Khách hàng vs Doanh thu theo Cụm", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8.5, frameon=True)

    bars_rep = axes[1].bar(
        labels_x,
        prof_indexed["repurchase_rate_90d"] * 100,
        color=[PALETTE[1], PALETTE[2], PALETTE[0], NOISE_COLOR],
        width=0.52,
    )
    for bar, (_, row) in zip(bars_rep, prof_indexed.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.8,
            f"{row['repurchase_rate_90d']*100:.1f}%\n(M_med=£{row['monetary_median']:,.0f})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[1].set_ylim(0, 102)
    axes[1].set_ylabel("Tỷ lệ mua lại trong 90 ngày (%)")
    axes[1].set_title("Hành vi mua lại 90 ngày và Chi tiêu trung vị", fontsize=11, fontweight="bold")

    # Phân rã 4 nhóm con trong Noise (-1)
    short_noise_names = [
        "1. Siêu VIP\n(F>=5, M>=P90)",
        "2. Mua lô lớn\n(F<5, M>=P90)",
        "3. Chi tiêu nhỏ\n(M < P25)",
        "4. Lệch biên\n(R=1 / khác)",
    ]
    axes[2].barh(
        short_noise_names,
        noise_summary_df["customers"],
        color=["#b91c1c", "#f97316", "#64748b", "#0ea5e9"],
    )
    for idx, row in noise_summary_df.iterrows():
        axes[2].text(
            row["customers"] + 1.2,
            idx,
            f"n={int(row['customers'])} | Mua lại: {row['repurchase_rate_90d']*100:.1f}%\nDT 90d: £{row['future_revenue_total']:,.0f}",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    axes[2].set_xlim(0, max(noise_summary_df["customers"]) * 1.68)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Số lượng khách hàng trong tập Noise (-1)")
    axes[2].set_title("Giải phẫu 4 nhóm con bên trong tập Nhiễu (-1)", fontsize=11, fontweight="bold")

    fig.suptitle("Hình 5: Phân tích hồ sơ các cụm DBSCAN và giải phẫu tập khách hàng Nhiễu (Noise = -1)", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    p5 = FIGURES_DIR / "dbscan_05_cluster_and_noise_profiles.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p5)

    # 6. Hình 6: Đánh giá khả năng phát hiện cụm có hình dạng khác nhau (RFM thực tế + 4 cấu trúc chuẩn)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    km_rfm = shape_extra["km_rfm_labels"]

    # Panel (0,0): DBSCAN trên mặt phẳng (Frequency_scaled, Monetary_scaled)
    for cid in sorted(dev_df["cluster"].unique()):
        sub = dev_df["cluster"] == cid
        axes[0, 0].scatter(
            dev_df.loc[sub, "Frequency_scaled"],
            dev_df.loc[sub, "Monetary_scaled"],
            s=16 if cid == -1 else 9,
            alpha=0.8 if cid == -1 else 0.5,
            color=cluster_colors.get(int(cid), "#64748b"),
            marker="x" if cid == -1 else "o",
            label=cluster_short.get(int(cid), str(cid)),
        )
    axes[0, 0].set_title("Online Retail RFM: DBSCAN\n(Tách đúng các dải phẳng F=1, F=2, ống F>=3 & Nhiễu)", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("Frequency_scaled")
    axes[0, 0].set_ylabel("Monetary_scaled")
    axes[0, 0].legend(loc="lower right", fontsize=7.5, frameon=True)

    # Panel (0,1): K-Means (k=3) trên cùng mặt phẳng (Frequency_scaled, Monetary_scaled)
    for cid, color in zip([0, 1, 2], [PALETTE[0], PALETTE[1], PALETTE[2]]):
        sub = km_rfm == cid
        axes[0, 1].scatter(
            dev_df.loc[sub, "Frequency_scaled"],
            dev_df.loc[sub, "Monetary_scaled"],
            s=9,
            alpha=0.5,
            color=color,
            label=f"K-Means Cụm {cid} (n={int(sub.sum())})",
        )
    axes[0, 1].set_title("Online Retail RFM: K-Means (k=3)\n(Cắt ngang qua các dải tần suất & nuốt trọn ngoại lệ)", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Frequency_scaled")
    axes[0, 1].set_ylabel("Monetary_scaled")
    axes[0, 1].legend(loc="lower right", fontsize=8, frameon=True)

    # 4 kịch bản hình học chuẩn ở 4 ô còn lại
    target_axes = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, (b_name, (X_b, y_true, y_db, y_km)) in zip(target_axes, shape_extra["bench_plot_data"].items()):
        valid_t = y_true != -1
        y_db_eval = y_db[valid_t].copy()
        noise_pos = np.flatnonzero(y_db_eval == -1)
        if len(noise_pos) > 0:
            y_db_eval[noise_pos] = -np.arange(1, len(noise_pos) + 1)
        ari_db = adjusted_rand_score(y_true[valid_t], y_db_eval)
        ari_km = adjusted_rand_score(y_true[valid_t], y_km[valid_t])
        for cid in sorted(np.unique(y_db)):
            mask = y_db == cid
            col = NOISE_COLOR if cid == -1 else PALETTE[int(cid) % len(PALETTE)]
            ax.scatter(
                X_b[mask, 0],
                X_b[mask, 1],
                s=18 if cid == -1 else 14,
                alpha=0.75,
                color=col,
                marker="x" if cid == -1 else "o",
            )
        ax.set_title(f"{b_name}\nDBSCAN ARI = {ari_db:.3f} vs K-Means ARI = {ari_km:.3f}", fontsize=9.8, fontweight="bold")
        ax.set_xlabel("Chiều đặc trưng 1")
        ax.set_ylabel("Chiều đặc trưng 2")

    fig.suptitle("Hình 6: Đánh giá khả năng phát hiện cụm có hình dạng khác nhau (DBSCAN vs K-Means)", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    p6 = FIGURES_DIR / "dbscan_06_shape_capability_comparison.png"
    fig.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p6)

    return saved_paths


def export_team_outputs(
    data_dir: Path,
    outputs_root: Path,
    chosen_eps: float,
    chosen_ms: int,
) -> dict[str, dict]:
    """Chạy DBSCAN với bộ siêu tham số đã chốt trên cả Development và Final và lưu kết quả theo chuẩn dự án."""
    results = {}
    for phase in ["development", "final"]:
        rfm, feat, outcomes = load_phase(data_dir, phase)
        X = feat[FEATURE_COLUMNS].to_numpy(dtype=float)

        t0 = time.perf_counter()
        model = DBSCAN(eps=chosen_eps, min_samples=chosen_ms, metric="euclidean")
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - t0

        phase_out_dir = outputs_root / phase
        phase_out_dir.mkdir(parents=True, exist_ok=True)

        pred_df = pd.DataFrame({
            "CustomerID": feat["CustomerID"].astype("string"),
            "cluster": pd.Series(labels, dtype="int64").astype("string"),
        })
        csv_path = phase_out_dir / "dbscan.csv"
        json_path = phase_out_dir / "dbscan.json"
        pred_df.to_csv(csv_path, index=False)

        metadata = {
            "model": "dbscan",
            "phase": phase,
            "runtime_seconds": round(float(elapsed), 6),
            "parameters": {
                "eps": chosen_eps,
                "min_samples": chosen_ms,
                "metric": "euclidean",
                "algorithm": "auto",
            },
        }
        json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        ev = evaluate_labels(X, labels, model_name="dbscan", runtime_seconds=elapsed)
        pol_default = evaluate_cluster_priority(rfm, outcomes, pred_df, "dbscan")

        # Đánh giá bổ sung: Chính sách có tách riêng nhóm VIP trong Noise (-1)
        # (Chỉ dùng Monetary/Frequency/Recency quá khứ, hoàn toàn không dùng dữ liệu tương lai)
        joined = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]].merge(
            pred_df, on="CustomerID", validate="one_to_one"
        ).merge(
            outcomes[["CustomerID", "repurchased_in_90d", "future_revenue"]],
            on="CustomerID", validate="one_to_one",
        )
        q90_past_m = float(joined["Monetary"].quantile(0.90))
        joined["effective_group"] = np.where(
            (joined["cluster"] == "-1") & (joined["Monetary"] >= q90_past_m) & (joined["Frequency"] >= 5),
            "VIP_NOISE",
            joined["cluster"],
        )
        vip_lbl_df = pd.DataFrame({"CustomerID": joined["CustomerID"], "cluster": joined["effective_group"]})
        pol_vip_aware = evaluate_cluster_priority(rfm, outcomes, vip_lbl_df, "dbscan_vip_aware")

        results[phase] = {
            "evaluation": ev,
            "policy_default": pol_default,
            "policy_vip_aware": pol_vip_aware,
            "csv_path": str(csv_path),
            "json_path": str(json_path),
        }
    return results


def main() -> None:
    data_dir = ROOT / "data_cleaned"
    outputs_root = ROOT / "models" / "outputs"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Đang đọc dữ liệu Development và phân tích khoảng cách k-distance...")
    dev_rfm, dev_feat, dev_outcomes = load_phase(data_dir, "development")
    X_dev = dev_feat[FEATURE_COLUMNS].to_numpy(dtype=float)

    min_samples_list = [4, 5, 6, 8, 10, 12, 15, 20]
    eps_list = [round(x, 2) for x in np.arange(0.20, 0.75, 0.05)]

    k_dist_df, k_curves = compute_k_distance_summary(X_dev, min_samples_list)
    k_dist_df.to_csv(TABLES_DIR / "dbscan_k_distance_summary.csv", index=False)

    print("2. Đang thực hiện Grid Search trên lưới (eps, min_samples)...")
    grid_df = run_hyperparameter_grid(X_dev, dev_rfm, dev_outcomes, eps_list, min_samples_list)
    grid_df.to_csv(TABLES_DIR / "dbscan_grid_search_development.csv", index=False)

    # Tham số tối ưu được chọn trên Development: eps = 0.45, min_samples = 8
    chosen_eps = 0.45
    chosen_ms = 8
    best_labels_dev = DBSCAN(eps=chosen_eps, min_samples=chosen_ms, metric="euclidean").fit_predict(X_dev)

    print("3. Đang phân tích hồ sơ các cụm và bóc tách khách hàng Nhiễu (Noise = -1)...")
    dev_full_df, dev_cluster_prof, dev_noise_prof = analyze_clusters_and_noise(
        dev_rfm, dev_feat, dev_outcomes, best_labels_dev
    )
    dev_cluster_prof.to_csv(TABLES_DIR / "dbscan_development_cluster_profiles.csv", index=False)
    dev_noise_prof.to_csv(TABLES_DIR / "dbscan_development_noise_breakdown.csv", index=False)

    print("4. Đang đánh giá khả năng phát hiện cụm có hình dạng khác nhau...")
    rfm_geom_df, bench_df, shape_extra = evaluate_cluster_geometry_and_shapes(X_dev, best_labels_dev)
    rfm_geom_df.to_csv(TABLES_DIR / "dbscan_rfm_cluster_geometry.csv", index=False)
    bench_df.to_csv(TABLES_DIR / "dbscan_shape_benchmarks.csv", index=False)

    print("5. Đang tạo các biểu đồ phân tích...")
    saved_figs = generate_all_figures(
        dev_full_df,
        X_dev,
        k_dist_df,
        k_curves,
        grid_df,
        dev_cluster_prof,
        dev_noise_prof,
        shape_extra,
        chosen_eps,
        chosen_ms,
    )

    print("6. Đang xuất nhãn cụm cho Development và Final...")
    phase_results = export_team_outputs(data_dir, outputs_root, chosen_eps, chosen_ms)

    # Lưu bảng hồ sơ Final
    fin_rfm, fin_feat, fin_outcomes = load_phase(data_dir, "final")
    X_fin = fin_feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    best_labels_fin = DBSCAN(eps=chosen_eps, min_samples=chosen_ms, metric="euclidean").fit_predict(X_fin)
    _, fin_cluster_prof, fin_noise_prof = analyze_clusters_and_noise(
        fin_rfm, fin_feat, fin_outcomes, best_labels_fin
    )
    fin_cluster_prof.to_csv(TABLES_DIR / "dbscan_final_cluster_profiles.csv", index=False)
    fin_noise_prof.to_csv(TABLES_DIR / "dbscan_final_noise_breakdown.csv", index=False)

    print("\n================ TÓM TẮT KẾT QUẢ MÔ HÌNH DBSCAN ================")
    print(f"Tham số tối ưu (chốt trên Development): eps = {chosen_eps}, min_samples = {chosen_ms}")
    print("\n--- Bảng k-distance Knee Summary ---")
    print(k_dist_df[["min_samples", "p85", "p90", "p95", "knee_quantile_full", "knee_eps_full", "knee_eps_trim995"]].to_string(index=False))
    print("\n--- Các cấu hình đạt chuẩn Clean Macro Regime trên Development ---")
    clean_configs = grid_df.loc[grid_df["clean_macro_regime"]].sort_values(
        ["silhouette", "calinski_harabasz"], ascending=[False, False]
    )
    print(clean_configs[[
        "eps", "min_samples", "n_clusters", "noise_customers", "noise_fraction",
        "silhouette", "davies_bouldin", "calinski_harabasz",
        "dev_selected_repurchase_rate_15pct", "dev_future_revenue_share_15pct"
    ]].to_string(index=False))

    print("\n--- Hồ sơ cụm trên Development (eps=0.45, min_samples=8) ---")
    print(dev_cluster_prof.to_string(index=False))

    print("\n--- Giải phẫu nhóm Nhiễu (Noise = -1) trên Development ---")
    print(dev_noise_prof.to_string(index=False))

    print("\n--- Đặc trưng hình học các cụm trên không gian RFM 3D ---")
    print(rfm_geom_df.to_string(index=False))

    print("\n--- Kết quả kiểm thử khả năng phát hiện cụm có hình dạng khác nhau ---")
    print(bench_df.to_string(index=False))

    print("\n--- Kết quả đánh giá trên Development và Final ---")
    for phase, info in phase_results.items():
        ev = info["evaluation"]
        p_def = info["policy_default"]
        p_vip = info["policy_vip_aware"]
        print(
            f"[{phase.upper()}] Khách hàng={ev['customers']:,} | Số cụm={ev['n_clusters']} | "
            f"Nhiễu={ev['noise_customers']} ({ev['noise_fraction']:.2%}) | "
            f"Silhouette={ev['silhouette']:.4f} | DBI={ev['davies_bouldin']:.4f} | CHI={ev['calinski_harabasz']:.1f}"
        )
        print(
            f"   -> Chính sách 15% mặc định (xếp Noise cuối): Mua lại={p_def['selected_repurchase_rate']:.2%}, "
            f"Tỷ trọng DT tương lai={p_def['future_revenue_share']:.2%}"
        )
        print(
            f"   -> Chính sách 15% có tách VIP-Noise quá khứ: Mua lại={p_vip['selected_repurchase_rate']:.2%}, "
            f"Tỷ trọng DT tương lai={p_vip['future_revenue_share']:.2%}"
        )
    print("\nĐã lưu các biểu đồ:", [p.name for p in saved_figs])


if __name__ == "__main__":
    main()
