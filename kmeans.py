"""
Xây dựng và đánh giá mô hình phân cụm K-Means cho bài toán Phân khúc Khách hàng Online Retail.

Nhiệm vụ:
1. Xây dựng K-Means bằng scikit-learn trên ba đặc trưng RFM đã chuẩn hóa dùng chung của nhóm.
2. Tìm số cụm k tối ưu trên tập Development bằng Elbow Method (Kneedle trên inertia) và Silhouette Score,
   kèm Davies-Bouldin, Calinski-Harabasz (từ evaluator chung) và độ ổn định bootstrap (ARI).
3. Trực quan hóa các cụm khách hàng (Elbow/Silhouette, silhouette diagram, PCA 2D/3D, tâm cụm, boxplot RFM).
4. Phân tích đặc điểm từng cụm bằng RFM gốc và đặt tên cụm theo hồ sơ, không theo số nhãn.
5. Xuất nhãn cụm cho Development và Final theo hợp đồng dữ liệu (`docs/team_data_guide.md`).

Nguyên tắc: chọn k chỉ trên Development bằng chỉ số nội tại; outcome 90 ngày của Development chỉ dùng để mô tả
cụm sau khi đã chọn k. Script không đọc outcome của Final; phần đó do `scripts/evaluate_temporal.py` đánh giá.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retail_segmentation.evaluation import evaluate_labels  # noqa: E402
from retail_segmentation.pipeline import FEATURE_COLUMNS, RFM_COLUMNS  # noqa: E402
from retail_segmentation.temporal_evaluation import load_phase  # noqa: E402

DATA_DIR = ROOT / "data_cleaned"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"
OUTPUTS_ROOT = ROOT / "models" / "outputs"

K_VALUES = tuple(range(2, 11))
RANDOM_STATE = 42
N_INIT = 20
MAX_ITER = 300
STABILITY_SEEDS = tuple(range(10))
STABILITY_SAMPLE_FRACTION = 0.8
MIN_CLUSTER_SHARE = 0.05
MIN_STABILITY_ARI = 0.90
CHOICE_RANGE = (3, 6)
SILHOUETTE_TOLERANCE = 0.02

PALETTE = ["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#64748b", "#a855f7"]


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------

def load_model_inputs(data_dir: Path, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc RFM gốc và feature của một phase mà KHÔNG đọc outcome (dùng cho Final)."""
    phase_dir = data_dir / "temporal" / phase
    rfm = pd.read_csv(phase_dir / "rfm_customers.csv", dtype={"CustomerID": "string"})
    feat = pd.read_csv(phase_dir / "rfm_model_features.csv", dtype={"CustomerID": "string"})
    for name, table in (("rfm", rfm), ("features", feat)):
        if table["CustomerID"].isna().any() or table["CustomerID"].duplicated().any():
            raise ValueError(f"{phase} {name} has missing or duplicate CustomerID")
    if set(rfm["CustomerID"]) != set(feat["CustomerID"]):
        raise ValueError(f"{phase}: RFM and feature customer sets differ")
    if not set(FEATURE_COLUMNS) <= set(feat.columns):
        raise ValueError(f"{phase}: shared features are incomplete")
    if {"repurchased_in_90d", "future_revenue", "future_invoices"} & set(feat.columns):
        raise ValueError(f"{phase}: future outcomes leaked into model features")
    return rfm, feat


def feature_matrix(feat: pd.DataFrame) -> np.ndarray:
    """Ma trận X đúng ba cột `_scaled` theo thứ tự hợp đồng; không chuẩn hóa lại."""
    X = feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains missing or non-finite values")
    return X


def load_scaler(data_dir: Path, phase: str):
    """Scaler đã fit trên log1p(RFM) của phase, dùng để đổi tâm cụm về đơn vị gốc."""
    return joblib.load(data_dir / "temporal" / phase / "preprocessor.joblib")["scaler"]


# ---------------------------------------------------------------------------
# Mô hình và chọn k
# ---------------------------------------------------------------------------

def kmeans_parameters(k: int, random_state: int = RANDOM_STATE) -> dict:
    return {
        "n_clusters": int(k),
        "init": "k-means++",
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "random_state": int(random_state),
        "algorithm": "lloyd",
    }


def fit_kmeans(X: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> tuple[KMeans, np.ndarray, float]:
    """Fit K-Means với cấu hình chung; trả model, nhãn và thời gian chạy (giây)."""
    model = KMeans(**kmeans_parameters(k, random_state))
    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    return model, labels, time.perf_counter() - t0


def stability_ari(
    X: np.ndarray,
    k: int,
    reference_labels: np.ndarray,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
    sample_fraction: float = STABILITY_SAMPLE_FRACTION,
) -> tuple[float, float]:
    """Độ ổn định bootstrap: fit trên 80% khách ngẫu nhiên, gán lại toàn bộ, so ARI với nhãn tham chiếu."""
    scores = []
    n_sample = max(k + 1, int(round(len(X) * sample_fraction)))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_sample, replace=False)
        model = KMeans(**kmeans_parameters(k, random_state=seed)).fit(X[idx])
        scores.append(adjusted_rand_score(reference_labels, model.predict(X)))
    return float(np.mean(scores)), float(np.min(scores))


def search_k(X: np.ndarray, k_values: tuple[int, ...] = K_VALUES, with_stability: bool = True) -> pd.DataFrame:
    """Quét k: inertia (Elbow), Silhouette/DBI/CHI từ evaluator chung, quy mô cụm, độ ổn định, thời gian."""
    rows = []
    for k in k_values:
        model, labels, runtime = fit_kmeans(X, k)
        ev = evaluate_labels(X, labels, model_name=f"kmeans_k{k}", runtime_seconds=runtime)
        sizes = np.array(list(ev["cluster_sizes"].values()))
        ari_mean, ari_min = stability_ari(X, k, labels) if with_stability else (np.nan, np.nan)
        rows.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": ev["silhouette"],
            "davies_bouldin": ev["davies_bouldin"],
            "calinski_harabasz": ev["calinski_harabasz"],
            "min_cluster_share": float(sizes.min() / len(X)),
            "largest_cluster_share": float(sizes.max() / len(X)),
            "ari_mean": ari_mean,
            "ari_min": ari_min,
            "n_iter": int(model.n_iter_),
            "runtime_seconds": round(float(runtime), 5),
            "cluster_sizes": json.dumps(ev["cluster_sizes"]),
            "parameters": json.dumps(kmeans_parameters(k)),
        })
    return pd.DataFrame(rows)


def find_elbow_k(k_values, inertia) -> int:
    """Kneedle cho đường inertia giảm lồi: k xa nhất bên dưới dây cung nối hai đầu (sau chuẩn hóa [0, 1])."""
    k_arr = np.asarray(k_values, dtype=float)
    y = np.asarray(inertia, dtype=float)
    if len(k_arr) < 3:
        return int(k_arr[0])
    x_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    span = y.max() - y.min()
    if span <= 0:
        return int(k_arr[0])
    y_norm = (y - y.min()) / span
    distance_below_chord = (1.0 - x_norm) - y_norm
    return int(k_arr[int(np.argmax(distance_below_chord))])


def choose_k(search_df: pd.DataFrame, elbow_k: int) -> tuple[int, str]:
    """Quy tắc chọn k cố định, chỉ dùng chỉ số nội tại của Development.

    Trong CHOICE_RANGE, giữ k có cụm nhỏ nhất >= MIN_CLUSTER_SHARE và ARI bootstrap trung bình >= MIN_STABILITY_ARI;
    chọn Silhouette cao nhất. Nếu k của Elbow cũng hợp lệ và Silhouette chỉ thấp hơn < SILHOUETTE_TOLERANCE
    thì ưu tiên k của Elbow (hai tiêu chí đồng thuận).
    """
    lo, hi = CHOICE_RANGE
    in_range = search_df.loc[search_df["k"].between(lo, hi)]
    valid = in_range.loc[
        in_range["min_cluster_share"].ge(MIN_CLUSTER_SHARE)
        & (in_range["ari_mean"].ge(MIN_STABILITY_ARI) | in_range["ari_mean"].isna())
    ]
    note = ""
    if valid.empty:
        valid = in_range
        note = " Không k nào đạt đủ điều kiện quy mô/ổn định nên xét toàn bộ khoảng."
    best = valid.loc[valid["silhouette"].idxmax()]
    chosen = int(best["k"])
    reason = (
        f"Trong k∈[{lo},{hi}] có cụm nhỏ nhất ≥ {MIN_CLUSTER_SHARE:.0%} và ARI bootstrap ≥ {MIN_STABILITY_ARI}, "
        f"Silhouette cao nhất ở k={chosen} ({best['silhouette']:.4f}); Elbow (Kneedle) ở k={elbow_k}."
    )
    if elbow_k != chosen and elbow_k in set(valid["k"]):
        elbow_sil = float(valid.loc[valid["k"].eq(elbow_k), "silhouette"].iloc[0])
        if best["silhouette"] - elbow_sil < SILHOUETTE_TOLERANCE:
            chosen = int(elbow_k)
            reason += (
                f" Silhouette của k={elbow_k} ({elbow_sil:.4f}) chỉ kém < {SILHOUETTE_TOLERANCE}, "
                f"nên chọn k={elbow_k} để Elbow và Silhouette đồng thuận."
            )
    return chosen, reason + note


def relabel_by_value(centers: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Đánh số lại cụm theo điểm giá trị tâm cụm (-R + F + M, thang scaled); cụm 0 có giá trị cao nhất.

    K-Means đánh số cụm tùy ý theo seed/phase; sắp lại giúp cùng một ý nghĩa cụm có cùng số ở mọi lần chạy.
    """
    centers = np.asarray(centers, dtype=float)
    score = -centers[:, 0] + centers[:, 1] + centers[:, 2]
    order = np.argsort(-score, kind="stable")
    mapping = np.empty_like(order)
    mapping[order] = np.arange(len(order))
    return mapping[np.asarray(labels)], centers[order]


def fit_chosen_model(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit k đã chốt, đánh số lại theo giá trị; trả nhãn, tâm cụm, thời gian chạy, inertia."""
    model, labels, runtime = fit_kmeans(X, k)
    labels, centers = relabel_by_value(model.cluster_centers_, labels)
    return labels, centers, runtime, float(model.inertia_)


# ---------------------------------------------------------------------------
# Hồ sơ và đặt tên cụm
# ---------------------------------------------------------------------------

def centroids_original_units(centers: np.ndarray, scaler) -> pd.DataFrame:
    """Tâm cụm ở thang scaled và đổi về đơn vị gốc: expm1(inverse_transform(center))."""
    original = np.expm1(scaler.inverse_transform(centers))
    df = pd.DataFrame(centers, columns=FEATURE_COLUMNS)
    df.insert(0, "cluster", np.arange(len(centers)))
    df["Recency_days"] = original[:, 0].round(1)
    df["Frequency_invoices"] = original[:, 1].round(2)
    df["Monetary_gbp"] = original[:, 2].round(2)
    return df


def label_frame(customer_ids: pd.Series, labels: np.ndarray, rfm: pd.DataFrame) -> pd.DataFrame:
    """Ghép nhãn với RFM gốc theo CustomerID (không dựa vào thứ tự dòng)."""
    labeled = pd.DataFrame({"CustomerID": customer_ids.astype("string").to_numpy(), "cluster": np.asarray(labels, dtype=int)})
    return labeled.merge(rfm, on="CustomerID", validate="one_to_one")


def profile_clusters(labeled: pd.DataFrame, outcomes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hồ sơ từng cụm bằng RFM gốc; outcome (chỉ Development) được thêm để mô tả, không dùng chọn k."""
    df = labeled.copy()
    if outcomes is not None:
        df = df.merge(
            outcomes[["CustomerID", "repurchased_in_90d", "future_revenue"]], on="CustomerID", validate="one_to_one"
        )
    total_m = float(df["Monetary"].sum())
    rows = []
    for cid, grp in df.groupby("cluster", sort=True):
        countries = grp["PrimaryCountry"].value_counts(normalize=True)
        row = {
            "cluster": int(cid),
            "customers": int(len(grp)),
            "customer_share": round(len(grp) / len(df), 4),
            "recency_median": float(grp["Recency"].median()),
            "recency_iqr": float(grp["Recency"].quantile(.75) - grp["Recency"].quantile(.25)),
            "frequency_median": float(grp["Frequency"].median()),
            "frequency_mean": round(float(grp["Frequency"].mean()), 2),
            "single_purchase_share": round(float(grp["Frequency"].eq(1).mean()), 4),
            "monetary_median": round(float(grp["Monetary"].median()), 2),
            "monetary_mean": round(float(grp["Monetary"].mean()), 2),
            "monetary_total": round(float(grp["Monetary"].sum()), 2),
            "monetary_share": round(float(grp["Monetary"].sum()) / total_m, 4),
            "top_country": countries.index[0] if len(countries) else None,
            "top_country_share": round(float(countries.iloc[0]), 4) if len(countries) else np.nan,
        }
        if outcomes is not None:
            total_future = float(df["future_revenue"].sum())
            row.update({
                "repurchase_rate_90d": round(float(grp["repurchased_in_90d"].mean()), 4),
                "future_revenue_mean": round(float(grp["future_revenue"].mean()), 2),
                "future_revenue_share": round(float(grp["future_revenue"].sum()) / total_future, 4)
                if total_future > 0 else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def name_clusters(profile: pd.DataFrame, rfm: pd.DataFrame) -> dict[int, str]:
    """Đặt tên cụm bằng cách so median của cụm với median toàn tập (tên phụ thuộc hồ sơ, không phụ thuộc số nhãn)."""
    r_med, f_med, m_med = (float(rfm[col].median()) for col in RFM_COLUMNS)
    m_p90 = float(rfm["Monetary"].quantile(.90))
    base = {}
    for _, row in profile.iterrows():
        recent = row["recency_median"] <= r_med
        frequent = row["frequency_median"] > f_med
        high_m = row["monetary_median"] > m_med
        if recent and frequent and row["monetary_median"] > m_p90:
            name = "Trung thành VIP (chi tiêu rất cao)"
        elif recent and frequent and high_m:
            name = "Trung thành giá trị cao"
        elif recent and frequent:
            name = "Mua đều, chi tiêu vừa"
        elif recent and high_m:
            name = "Mới mua gần đây, đơn giá trị cao"
        elif recent:
            name = "Khách mới / mua gần đây"
        elif frequent or high_m:
            name = "Có nguy cơ rời bỏ (từng giá trị cao)"
        else:
            name = "Ngủ đông / mua ít"
        base[int(row["cluster"])] = (name, float(row["monetary_median"]))
    names = {}
    for name in {n for n, _ in base.values()}:
        members = sorted((c for c, (n, _) in base.items() if n == name), key=lambda c: -base[c][1])
        for rank, cid in enumerate(members, start=1):
            suffix = f" – mức {rank}" if len(members) > 1 else ""
            names[cid] = f"Cụm {cid}: {name}{suffix}"
    return dict(sorted(names.items()))


# ---------------------------------------------------------------------------
# Trực quan hóa
# ---------------------------------------------------------------------------

def _colors(n: int) -> list[str]:
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _save(fig: plt.Figure, name: str, figures_dir: Path) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_k_selection(search_df: pd.DataFrame, elbow_k: int, chosen_k: int, figures_dir: Path = FIGURES_DIR) -> Path:
    """Hình 1: Elbow (inertia), Silhouette, Davies-Bouldin và độ ổn định ARI theo k."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    k = search_df["k"]
    panels = [
        (axes[0, 0], "inertia", "Inertia (WCSS)", "Elbow Method", elbow_k, "Elbow"),
        (axes[0, 1], "silhouette", "Silhouette (cao hơn tốt hơn)", "Silhouette Score", chosen_k, "k chọn"),
        (axes[1, 0], "davies_bouldin", "Davies-Bouldin (thấp hơn tốt hơn)", "Davies-Bouldin Index", chosen_k, "k chọn"),
    ]
    for ax, col, ylabel, title, mark_k, mark_label in panels:
        ax.plot(k, search_df[col], marker="o", color=PALETTE[0])
        mark = search_df.loc[search_df["k"].eq(mark_k), col]
        if not mark.empty:
            ax.scatter([mark_k], [mark.iloc[0]], s=160, facecolors="none", edgecolors="#dc2626", linewidths=2,
                       label=f"{mark_label}: k={mark_k}", zorder=3)
            ax.legend()
        ax.set(xlabel="Số cụm k", ylabel=ylabel, title=title, xticks=list(k))
    ax = axes[1, 1]
    ax.plot(k, search_df["ari_mean"], marker="o", color=PALETTE[1], label="ARI trung bình")
    ax.plot(k, search_df["ari_min"], marker="s", linestyle="--", color=PALETTE[2], label="ARI nhỏ nhất")
    ax.axhline(MIN_STABILITY_ARI, color="#64748b", linestyle=":", label=f"Ngưỡng {MIN_STABILITY_ARI}")
    ax.axvline(chosen_k, color="#dc2626", alpha=.4)
    ax.set(xlabel="Số cụm k", ylabel="ARI so với nhãn đầy đủ", title="Độ ổn định bootstrap (80% khách)",
           xticks=list(k), ylim=(min(.5, float(search_df["ari_min"].min()) - .05), 1.02))
    ax.legend()
    fig.suptitle("Chọn số cụm K-Means trên Development", y=1.01)
    return _save(fig, "kmeans_01_elbow_silhouette.png", figures_dir)


def plot_silhouette_diagrams(X: np.ndarray, k_values: list[int], figures_dir: Path = FIGURES_DIR) -> Path:
    """Hình 2: silhouette diagram từng khách cho các k ứng viên quanh k đã chọn."""
    fig, axes = plt.subplots(1, len(k_values), figsize=(4.6 * len(k_values), 4.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, k_values):
        model, labels, _ = fit_kmeans(X, k)
        labels, _ = relabel_by_value(model.cluster_centers_, labels)
        values = silhouette_samples(X, labels)
        y_lower = 10
        for cid, color in zip(range(k), _colors(k)):
            vals = np.sort(values[labels == cid])
            ax.fill_betweenx(np.arange(y_lower, y_lower + len(vals)), 0, vals, color=color, alpha=.8)
            ax.text(-0.08, y_lower + len(vals) / 2, str(cid), va="center", fontsize=9)
            y_lower += len(vals) + 10
        avg = float(values.mean())
        ax.axvline(avg, color="#dc2626", linestyle="--", label=f"TB = {avg:.3f}")
        ax.set(title=f"k = {k}", xlabel="Silhouette từng khách", yticks=[], xlim=(-0.2, 1))
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Khách hàng (xếp theo cụm)")
    fig.suptitle("Silhouette diagram trên Development", y=1.02)
    return _save(fig, "kmeans_02_silhouette_diagrams.png", figures_dir)


def plot_pca_clusters(
    X: np.ndarray, labels: np.ndarray, centers: np.ndarray, names: dict[int, str], figures_dir: Path = FIGURES_DIR
) -> tuple[Path, PCA]:
    """Hình 3: PCA 2D (kèm tâm cụm chiếu) và scatter 3D trên ba feature chuẩn hóa."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    X2 = pca.transform(X)
    C2 = pca.transform(centers)
    fig = plt.figure(figsize=(14, 5.6))
    ax = fig.add_subplot(1, 2, 1)
    for cid, color in zip(range(len(centers)), _colors(len(centers))):
        mask = labels == cid
        ax.scatter(X2[mask, 0], X2[mask, 1], s=10, alpha=.4, color=color, label=names.get(cid, f"Cụm {cid}"),
                   rasterized=True)
    ax.scatter(C2[:, 0], C2[:, 1], marker="X", s=220, color="black", edgecolors="white", linewidths=1.5,
               label="Tâm cụm")
    ev = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({ev[0]:.1%} phương sai)", ylabel=f"PC2 ({ev[1]:.1%} phương sai)",
           title=f"PCA 2D (giữ {ev.sum():.1%} phương sai)")
    ax.legend(fontsize=8, loc="best")
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    for cid, color in zip(range(len(centers)), _colors(len(centers))):
        mask = labels == cid
        ax3.scatter(X[mask, 0], X[mask, 1], X[mask, 2], s=9, alpha=.55, color=color, rasterized=True)
    ax3.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker="X", s=200, color="black", depthshade=False)
    ax3.set(xlabel="Recency_scaled", ylabel="Frequency_scaled", zlabel="Monetary_scaled", title="Không gian RFM 3D")
    return _save(fig, "kmeans_03_pca_clusters.png", figures_dir), pca


def plot_centroid_heatmap(centroids: pd.DataFrame, names: dict[int, str], figures_dir: Path = FIGURES_DIR) -> Path:
    """Hình 4: heatmap tâm cụm (màu = thang scaled, chữ = đơn vị gốc) và snake plot."""
    scaled = centroids.set_index("cluster")[FEATURE_COLUMNS]
    annot = np.array([
        [f"{r.Recency_days:.0f} ngày", f"{r.Frequency_invoices:.1f} HĐ", f"£{r.Monetary_gbp:,.0f}"]
        for r in centroids.itertuples()
    ])
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.8 * len(scaled) + 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    sns.heatmap(scaled, annot=annot, fmt="", cmap="RdBu_r", center=0, ax=axes[0], cbar_kws={"label": "Giá trị scaled"})
    axes[0].set_yticklabels([names.get(c, str(c)) for c in scaled.index], rotation=0)
    axes[0].set_xticklabels(["Recency", "Frequency", "Monetary"])
    axes[0].set(title="Tâm cụm (màu: scaled, chữ: đơn vị gốc)", ylabel="")
    for (cid, row), color in zip(scaled.iterrows(), _colors(len(scaled))):
        axes[1].plot(["Recency", "Frequency", "Monetary"], row.to_numpy(), marker="o", color=color,
                     label=names.get(cid, str(cid)))
    axes[1].axhline(0, color="#64748b", linewidth=.8)
    axes[1].set(title="Snake plot tâm cụm", ylabel="Giá trị scaled (0 = trung bình)")
    axes[1].legend(fontsize=8)
    return _save(fig, "kmeans_04_centroid_heatmap.png", figures_dir)


def plot_rfm_boxplots(labeled: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Hình 5: phân phối log1p(R/F/M) gốc theo cụm."""
    df = labeled.copy()
    k = int(df["cluster"].nunique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col in zip(axes, RFM_COLUMNS):
        df[f"log1p_{col}"] = np.log1p(df[col])
        sns.boxplot(data=df, x="cluster", y=f"log1p_{col}", hue="cluster", palette=_colors(k), legend=False, ax=ax,
                    fliersize=2)
        ax.set(title=col, xlabel="Cụm", ylabel=f"log1p({col})")
    fig.suptitle("Phân phối RFM gốc theo cụm (thang log1p)", y=1.03)
    return _save(fig, "kmeans_05_rfm_boxplots.png", figures_dir)


def plot_cluster_summary(profile: pd.DataFrame, names: dict[int, str], figures_dir: Path = FIGURES_DIR) -> Path:
    """Hình 6: tỷ lệ khách, tỷ trọng Monetary quá khứ và (nếu có) tỷ lệ mua lại 90 ngày theo cụm."""
    metrics = [("customer_share", "Tỷ lệ khách"), ("monetary_share", "Tỷ trọng Monetary quá khứ")]
    if "repurchase_rate_90d" in profile:
        metrics.append(("repurchase_rate_90d", "Tỷ lệ mua lại 90 ngày (mô tả)"))
    x = np.arange(len(profile))
    width = .8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(profile) + 3), 4.6))
    for i, ((col, label), color) in enumerate(zip(metrics, [PALETTE[0], PALETTE[2], PALETTE[1]])):
        bars = ax.bar(x + (i - (len(metrics) - 1) / 2) * width, profile[col] * 100, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
    ax.set_xticks(x, [names.get(int(c), str(c)).replace(": ", ":\n") for c in profile["cluster"]], fontsize=8)
    ax.set(ylabel="%", title="Quy mô, giá trị và hành vi mua lại theo cụm (Development)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "kmeans_06_cluster_summary.png", figures_dir)


# ---------------------------------------------------------------------------
# Xuất kết quả theo hợp đồng của nhóm
# ---------------------------------------------------------------------------

def export_labels(
    outputs_root: Path,
    phase: str,
    customer_ids: pd.Series,
    labels: np.ndarray,
    runtime_seconds: float,
    k: int,
    metrics: dict | None = None,
    selection: dict | None = None,
) -> tuple[Path, Path]:
    """Ghi `<phase>/kmeans.csv` (CustomerID,cluster; một dòng mỗi khách) và `kmeans.json` (tham số, thời gian chạy)."""
    if len(customer_ids) != len(labels):
        raise ValueError("customer_ids and labels differ in length")
    ids = pd.Series(customer_ids, dtype="string").reset_index(drop=True)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("customer_ids must be unique and non-null")
    phase_dir = outputs_root / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    csv_path = phase_dir / "kmeans.csv"
    json_path = phase_dir / "kmeans.json"
    pd.DataFrame({"CustomerID": ids, "cluster": np.asarray(labels, dtype=int)}).to_csv(csv_path, index=False)
    metadata = {
        "model": "kmeans",
        "phase": phase,
        "runtime_seconds": round(float(runtime_seconds), 6),
        "parameters": kmeans_parameters(k),
        "metrics": metrics or {},
        "selection": {**(selection or {}), "sklearn_version": sklearn.__version__},
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return csv_path, json_path


def _metric_summary(X: np.ndarray, labels: np.ndarray, runtime: float, inertia: float) -> dict:
    ev = evaluate_labels(X, labels, model_name="kmeans", runtime_seconds=runtime)
    return {
        "customers": ev["customers"],
        "n_clusters": ev["n_clusters"],
        "cluster_sizes": ev["cluster_sizes"],
        "inertia": round(inertia, 4),
        "silhouette": round(ev["silhouette"], 4),
        "davies_bouldin": round(ev["davies_bouldin"], 4),
        "calinski_harabasz": round(ev["calinski_harabasz"], 2),
    }


# ---------------------------------------------------------------------------
# Chạy toàn bộ
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="K-Means cho phân khúc khách hàng Online Retail")
    parser.add_argument("--k", type=int, help="Ghi đè k thay vì dùng quy tắc choose_k")
    parser.add_argument("--k-max", type=int, default=max(K_VALUES), help="k lớn nhất khi quét (mặc định 10)")
    parser.add_argument("--skip-figures", action="store_true", help="Không vẽ hình")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports", help="Nơi ghi tables/ và figures/")
    args = parser.parse_args()
    tables_dir = args.reports_dir / "tables"
    figures_dir = args.reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("1. Đọc Development và quét k...")
    dev_rfm, dev_feat, dev_outcomes = load_phase(args.data_dir, "development")
    X_dev = feature_matrix(dev_feat)
    search_df = search_k(X_dev, tuple(range(2, args.k_max + 1)))
    search_df.to_csv(tables_dir / "kmeans_k_search_development.csv", index=False)
    elbow_k = find_elbow_k(search_df["k"], search_df["inertia"])
    if args.k is not None:
        chosen_k, reason = args.k, f"k={args.k} do người chạy chỉ định (--k); Elbow ở k={elbow_k}."
    else:
        chosen_k, reason = choose_k(search_df, elbow_k)
    print(f"   {reason}")

    print("2. Fit k đã chốt trên Development, tạo hồ sơ cụm...")
    dev_labels, dev_centers, dev_runtime, dev_inertia = fit_chosen_model(X_dev, chosen_k)
    dev_labeled = label_frame(dev_feat["CustomerID"], dev_labels, dev_rfm)
    dev_profile = profile_clusters(dev_labeled, dev_outcomes)
    names = name_clusters(dev_profile, dev_rfm)
    dev_profile.insert(1, "cluster_name", dev_profile["cluster"].map(names))
    dev_profile.to_csv(tables_dir / "kmeans_development_cluster_profiles.csv", index=False)
    dev_centroids = centroids_original_units(dev_centers, load_scaler(args.data_dir, "development"))
    dev_centroids.insert(1, "cluster_name", dev_centroids["cluster"].map(names))
    dev_centroids.to_csv(tables_dir / "kmeans_centroids_development.csv", index=False)

    saved = []
    if not args.skip_figures:
        print("3. Vẽ biểu đồ...")
        saved.append(plot_k_selection(search_df, elbow_k, chosen_k, figures_dir))
        neighbours = [k for k in (chosen_k - 1, chosen_k, chosen_k + 1) if k >= 2]
        saved.append(plot_silhouette_diagrams(X_dev, neighbours, figures_dir))
        saved.append(plot_pca_clusters(X_dev, dev_labels, dev_centers, names, figures_dir)[0])
        saved.append(plot_centroid_heatmap(dev_centroids, names, figures_dir))
        saved.append(plot_rfm_boxplots(dev_labeled, figures_dir))
        saved.append(plot_cluster_summary(dev_profile, names, figures_dir))

    selection = {"k_values": search_df["k"].tolist(), "elbow_k": elbow_k, "chosen_k": chosen_k, "rule": reason}
    dev_metrics = _metric_summary(X_dev, dev_labels, dev_runtime, dev_inertia)
    export_labels(args.outputs_root, "development", dev_feat["CustomerID"], dev_labels, dev_runtime, chosen_k,
                  dev_metrics, selection)

    print("4. Chạy lại cùng cấu hình trên Final (không đọc outcome Final)...")
    fin_rfm, fin_feat = load_model_inputs(args.data_dir, "final")
    X_fin = feature_matrix(fin_feat)
    fin_labels, fin_centers, fin_runtime, fin_inertia = fit_chosen_model(X_fin, chosen_k)
    fin_labeled = label_frame(fin_feat["CustomerID"], fin_labels, fin_rfm)
    fin_profile = profile_clusters(fin_labeled)
    fin_names = name_clusters(fin_profile, fin_rfm)
    fin_profile.insert(1, "cluster_name", fin_profile["cluster"].map(fin_names))
    fin_profile.to_csv(tables_dir / "kmeans_final_cluster_profiles.csv", index=False)
    fin_centroids = centroids_original_units(fin_centers, load_scaler(args.data_dir, "final"))
    fin_centroids.insert(1, "cluster_name", fin_centroids["cluster"].map(fin_names))
    fin_centroids.to_csv(tables_dir / "kmeans_centroids_final.csv", index=False)
    fin_metrics = _metric_summary(X_fin, fin_labels, fin_runtime, fin_inertia)
    export_labels(args.outputs_root, "final", fin_feat["CustomerID"], fin_labels, fin_runtime, chosen_k,
                  fin_metrics, selection)

    pd.set_option("display.width", 200)
    print("\n================ TÓM TẮT K-MEANS ================")
    print(search_df[["k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz",
                     "min_cluster_share", "ari_mean", "runtime_seconds"]].round(4).to_string(index=False))
    print(f"\nElbow: k={elbow_k} | k chọn: {chosen_k}")
    for phase, metrics in (("development", dev_metrics), ("final", fin_metrics)):
        print(f"[{phase}] khách={metrics['customers']:,} | Silhouette={metrics['silhouette']:.4f} | "
              f"DBI={metrics['davies_bouldin']:.4f} | CHI={metrics['calinski_harabasz']:.1f} | "
              f"cỡ cụm={metrics['cluster_sizes']}")
    print("\n--- Hồ sơ cụm Development ---")
    print(dev_profile.drop(columns=["top_country"]).to_string(index=False))
    print("\n--- Hồ sơ cụm Final (chỉ RFM quá khứ) ---")
    print(fin_profile.drop(columns=["top_country"]).to_string(index=False))
    print(f"\nNhãn: {args.outputs_root / 'development' / 'kmeans.csv'}, {args.outputs_root / 'final' / 'kmeans.csv'}")
    if saved:
        print("Hình:", [p.name for p in saved])
    print("Tiếp theo: python scripts/evaluate_temporal.py --phase development --with-models")


if __name__ == "__main__":
    main()
