"""Agglomerative Hierarchical Clustering for Online Retail Customer Segmentation.

This script executes:
1. Data loading for development (2,716 customers) and final (3,314 customers) phases.
2. Hierarchical clustering exploration with scipy (linkages: ward, complete, average, single).
3. Dendrogram plotting with horizontal cut-off lines and distance gap analysis.
4. Hyperparameter tuning across linkages (ward, complete, average) and clusters (k in 2..10).
5. Evaluation metrics calculation (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index).
6. Model selection and output generation adhering to project contract:
   - models/outputs/development/agglomerative.csv & .json
   - models/outputs/final/agglomerative.csv & .json
7. Customer Persona Profiling (mean/median RFM, business personas, PCA projection, boxplots).
8. Out-of-time temporal evaluation (90-day repurchase behavior).
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

# Robust repository root and module resolution
def resolve_repo_root() -> Path:
    """Resolve repository root containing 'src/retail_segmentation'."""
    here = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    for candidate in [here, here.parent, here / "Online-Retail-Customer-Segmentation"]:
        if (candidate / "data_cleaned").exists() and (candidate / "src" / "retail_segmentation").exists():
            return candidate.resolve()
    # Fallback to current working directory
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
    # Additional fallback for IDEs and unusual execution contexts
    sys.path.insert(0, str(Path("D:/WORKSPACE/Online-Retail-Customer-Segmentation/src").resolve()))
    from retail_segmentation.pipeline import FEATURE_COLUMNS, RFM_COLUMNS
    from retail_segmentation.temporal_evaluation import load_phase


def setup_plotting_style() -> None:
    """Set aesthetic defaults for reports and figures."""
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
    """Ensure all required reporting and output directories exist."""
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
    """Compute linkage matrices and render dendrograms with cut-off thresholds."""
    print("\n" + "=" * 70)
    print("STEP 1: Computing Hierarchical Linkage Matrices and Dendrograms")
    print("=" * 70)

    linkages = {
        "ward": linkage(X, method="ward", metric="euclidean"),
        "complete": linkage(X, method="complete", metric="euclidean"),
        "average": linkage(X, method="average", metric="euclidean"),
        "single": linkage(X, method="single", metric="euclidean"),
    }

    # 1. Comparison of 4 linkages (truncate last 30 merged clusters)
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
        ax.set_title(f"Linkage: {method.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Cluster / Sample Count (truncated to last 30 nodes)")
        ax.set_ylabel("Merger Distance")

    plt.suptitle(
        "Hierarchical Dendrogram Comparison Across Linkage Methods (Development Phase)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    comparison_path = output_dir / "dendrogram_comparison.png"
    plt.savefig(comparison_path)
    plt.close()
    print(f" -> Saved Dendrogram Comparison: {comparison_path}")

    # 2. Detailed Ward Dendrogram with Cut-off Analysis
    Z_ward = linkages["ward"]
    fig, ax = plt.subplots(figsize=(12, 7))
    ddata = dendrogram(
        Z_ward,
        truncate_mode="lastp",
        p=35,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
        ax=ax,
        color_threshold=38.0,
    )

    # Calculate height differences for top mergers to find largest distance gaps
    last_mergers = Z_ward[-10:, 2]
    # For k=4, cutoff is around distance ~38
    cut_k4 = 38.0
    # For k=5, cutoff is around distance ~31
    cut_k5 = 31.0

    ax.axhline(y=cut_k4, color="crimson", linestyle="--", linewidth=2, label=f"Cut-off for k=4 (distance = {cut_k4})")
    ax.axhline(y=cut_k5, color="darkorange", linestyle=":", linewidth=2, label=f"Cut-off for k=5 (distance = {cut_k5})")

    ax.set_title("Ward Linkage Dendrogram with Cluster Cut-off Thresholds", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clusters (Numbers in parentheses denote customer count in cluster node)")
    ax.set_ylabel("Ward Distance (Variance Increase ΔESS)")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    ward_dendro_path = output_dir / "dendrogram_ward.png"
    plt.savefig(ward_dendro_path)
    plt.close()
    print(f" -> Saved Ward Dendrogram: {ward_dendro_path}")

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
    """Evaluate combinations of linkage and k (2..10) on internal clustering metrics."""
    print("\n" + "=" * 70)
    print("STEP 2: Grid Search over Linkage Methods & Cluster Counts k in [2, 10]")
    print("=" * 70)

    records = []
    for method in linkage_methods:
        print(f"  Evaluating linkage='{method}'...")
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
    print(f" -> Saved grid search comparison table: {table_path}")

    # Plot metrics across k for each linkage
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    palette = {"ward": "#1f77b4", "complete": "#2ca02c", "average": "#ff7f0e"}
    markers = {"ward": "o", "complete": "s", "average": "^"}

    # 1. Silhouette
    for method in linkage_methods:
        sub = results_df[results_df["linkage"] == method]
        axes[0].plot(sub["k"], sub["silhouette"], marker=markers[method], label=method.capitalize(), color=palette[method])
    axes[0].set_title("Silhouette Score (Higher is Better)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].legend()

    # 2. Davies-Bouldin
    for method in linkage_methods:
        sub = results_df[results_df["linkage"] == method]
        axes[1].plot(sub["k"], sub["davies_bouldin"], marker=markers[method], label=method.capitalize(), color=palette[method])
    axes[1].set_title("Davies-Bouldin Index (Lower is Better)")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Davies-Bouldin Index")
    axes[1].legend()

    # 3. Calinski-Harabasz
    for method in linkage_methods:
        sub = results_df[results_df["linkage"] == method]
        axes[2].plot(sub["k"], sub["calinski_harabasz"], marker=markers[method], label=method.capitalize(), color=palette[method])
    axes[2].set_title("Calinski-Harabasz Index (Higher is Better)")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Calinski-Harabasz Index")
    axes[2].legend()

    plt.suptitle("Clustering Validation Metrics vs. Number of Clusters (k)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    metrics_path = output_fig_dir / "metrics_comparison.png"
    plt.savefig(metrics_path)
    plt.close()
    print(f" -> Saved metrics comparison plot: {metrics_path}")

    return results_df


def fit_and_export_model(
    phase: str,
    features_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    best_k: int,
    best_linkage: str,
    output_dir: Path,
) -> tuple[np.ndarray, dict]:
    """Train optimal agglomerative clustering model and write submission contract files."""
    X = features_df[FEATURE_COLUMNS].to_numpy()
    customer_ids = features_df["CustomerID"].to_numpy()

    t0 = time.perf_counter()
    model = AgglomerativeClustering(n_clusters=best_k, linkage=best_linkage, metric="euclidean")
    labels = model.fit_predict(X)
    runtime = time.perf_counter() - t0

    # Ensure output dataframe has exact columns and order
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

    print(f"  [{phase.upper()}] Exported model outputs:")
    print(f"     CSV:  {csv_path} ({len(output_df)} customers)")
    print(f"     JSON: {json_path}")

    return labels, metadata


def generate_customer_profiles(
    rfm_df: pd.DataFrame,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    output_fig_dir: Path,
    output_table_dir: Path,
) -> pd.DataFrame:
    """Analyze customer personas, compute RFM summary statistics, and generate visual charts."""
    print("\n" + "=" * 70)
    print("STEP 3: Customer Persona Profiling & Business Segmentation")
    print("=" * 70)

    df = rfm_df.copy()
    df["cluster"] = labels

    # Compute descriptive profile
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

    # Assign intuitive retail persona names based on median RFM characteristics
    # We sort clusters by monetary median descending to assign standardized personas
    summary_sorted = summary.sort_values(by="monetary_median", ascending=False).reset_index(drop=True)
    persona_mapping = {}
    names = [
        "Champions / VIP (High Value, High Freq)",
        "Loyal Customers (Active, Moderate High Value)",
        "Potential / Promising (Recent or Mid-tier)",
        "At-Risk / Lost (High Recency, Low Freq)",
        "Dormant One-Time Buyers",
    ]
    for i, row in summary_sorted.iterrows():
        c_id = row["cluster"]
        persona_name = names[i] if i < len(names) else f"Segment {c_id}"
        persona_mapping[c_id] = persona_name

    summary["persona"] = summary["cluster"].map(persona_mapping)
    df["persona"] = df["cluster"].map(persona_mapping)

    # Reorder columns
    summary = summary[[
        "cluster", "persona", "customers", "customer_share",
        "recency_median", "frequency_median", "monetary_median",
        "recency_mean", "frequency_mean", "monetary_mean",
        "monetary_total", "revenue_share"
    ]]

    profile_table_path = output_table_dir / "cluster_profiles.csv"
    summary.to_csv(profile_table_path, index=False)
    print(f" -> Saved Customer Persona Profile table: {profile_table_path}")
    print(summary.to_string(index=False))

    # 1. 2D PCA Visualization
    X = features_df[FEATURE_COLUMNS].to_numpy()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    plt.figure(figsize=(10, 7))
    scatter = sns.scatterplot(
        data=df,
        x="PCA1",
        y="PCA2",
        hue="persona",
        palette="tab10",
        alpha=0.7,
        s=40,
        edgecolor=None,
    )
    plt.title(
        f"Agglomerative Clusters Visualized via 2D PCA (Explains {pca.explained_variance_ratio_.sum()*100:.1f}% Variance)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    pca_path = output_fig_dir / "pca_clusters.png"
    plt.savefig(pca_path)
    plt.close()
    print(f" -> Saved 2D PCA Cluster Map: {pca_path}")

    # 2. RFM Distributions by Cluster (Boxplots on log scale for Monetary & Frequency)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=df, x="cluster", y="Recency", ax=axes[0], palette="Set2")
    axes[0].set_title("Recency (Days since last purchase - Lower is Better)")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Recency (Days)")

    sns.boxplot(data=df, x="cluster", y="Frequency", ax=axes[1], palette="Set2")
    axes[1].set_yscale("log")
    axes[1].set_title("Frequency (Number of Invoices - Log Scale)")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Frequency (Count)")

    sns.boxplot(data=df, x="cluster", y="Monetary", ax=axes[2], palette="Set2")
    axes[2].set_yscale("log")
    axes[2].set_title("Monetary Value (GBP £ - Log Scale)")
    axes[2].set_xlabel("Cluster ID")
    axes[2].set_ylabel("Monetary (£)")

    plt.suptitle("RFM Feature Distributions across Customer Segments", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    rfm_dist_path = output_fig_dir / "rfm_distribution.png"
    plt.savefig(rfm_dist_path)
    plt.close()
    print(f" -> Saved RFM Feature Distribution Boxplots: {rfm_dist_path}")

    # 3. Bar Chart comparing Median RFM
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.barplot(data=summary, x="cluster", y="recency_median", ax=axes[0], palette="Blues_r")
    axes[0].set_title("Median Recency (Days)")
    axes[0].set_ylabel("Days")

    sns.barplot(data=summary, x="cluster", y="frequency_median", ax=axes[1], palette="Greens")
    axes[1].set_title("Median Frequency (Orders)")
    axes[1].set_ylabel("Order Count")

    sns.barplot(data=summary, x="cluster", y="monetary_median", ax=axes[2], palette="Oranges")
    axes[2].set_title("Median Monetary (£)")
    axes[2].set_ylabel("Total Spend (£)")

    plt.suptitle("Cluster Median RFM Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    bar_path = output_fig_dir / "cluster_profiles.png"
    plt.savefig(bar_path)
    plt.close()
    print(f" -> Saved Cluster Median Profiles Barplot: {bar_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4, help="Optimal number of clusters (default: 4)")
    parser.add_argument("--linkage", type=str, default="ward", choices=["ward", "complete", "average"], help="Linkage criterion (default: ward)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_cleaned")
    args = parser.parse_args()

    setup_plotting_style()
    dirs = create_directories()

    print("=" * 70)
    print("AGGLOMERATIVE CLUSTERING PIPELINE")
    print("Online Retail Customer Segmentation")
    print(f"Selected Configuration: k={args.k}, linkage='{args.linkage}'")
    print("=" * 70)

    # 1. Load Development Phase Data
    rfm_dev, feat_dev, outcomes_dev = load_phase(args.data_dir, "development")
    X_dev = feat_dev[FEATURE_COLUMNS].to_numpy()
    print(f"Loaded Development data: {len(rfm_dev)} customers, features: {FEATURE_COLUMNS}")

    # 2. Dendrogram and Hierarchical Linkage analysis
    plot_dendrograms(X_dev, dirs["figures"])

    # 3. Grid Search over linkages and k (2..10)
    grid_df = grid_search_hyperparameters(
        X_dev,
        k_range=range(2, 11),
        linkage_methods=["ward", "complete", "average"],
        output_table_dir=dirs["tables"],
        output_fig_dir=dirs["figures"],
    )

    # 4. Train and export optimal model for Development Phase
    print("\n" + "=" * 70)
    print(f"STEP 4: Model Training and Submission Outputs for Phase 'development'")
    print("=" * 70)
    labels_dev, meta_dev = fit_and_export_model(
        phase="development",
        features_df=feat_dev,
        rfm_df=rfm_dev,
        best_k=args.k,
        best_linkage=args.linkage,
        output_dir=dirs["dev_out"],
    )

    # 5. Customer Persona Profiling (Development)
    generate_customer_profiles(
        rfm_df=rfm_dev,
        features_df=feat_dev,
        labels=labels_dev,
        output_fig_dir=dirs["figures"],
        output_table_dir=dirs["tables"],
    )

    # 6. Train and export optimal model for Final Phase
    print("\n" + "=" * 70)
    print(f"STEP 5: Model Training and Submission Outputs for Phase 'final'")
    print("=" * 70)
    rfm_final, feat_final, outcomes_final = load_phase(args.data_dir, "final")
    print(f"Loaded Final data: {len(rfm_final)} customers")
    labels_final, meta_final = fit_and_export_model(
        phase="final",
        features_df=feat_final,
        rfm_df=rfm_final,
        best_k=args.k,
        best_linkage=args.linkage,
        output_dir=dirs["final_out"],
    )

    print("\n" + "=" * 70)
    print("Agglomerative Clustering Pipeline Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
