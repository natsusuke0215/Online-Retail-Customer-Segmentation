import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# --- TỰ ĐỘNG XÁC ĐỊNH THƯ MỤC GỐC DỰ ÁN (PROJECT ROOT) ---
# FILE_DIR là thư mục tests/, ROOT_DIR là thư mục gốc của dự án
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent

# Thêm thư mục gốc và src vào sys.path để import mô-đun nội bộ nếu cần
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

# --- ĐƯỜNG DẪN ĐẦU RA VÀ ĐẦU VÀO ---
DATA_PATH = ROOT_DIR / "data_cleaned" / "rfm_model_features.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
MODELS_DIR = ROOT_DIR / "models" / "outputs" / "final"

# Tạo thư mục nếu chưa tồn tại
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def test_or_run_gmm():
    print("==================================================")
    print("   CHẠY MÔ HÌNH GMM DẠNG TEST (THÀNH VIÊN 4)")
    print("==================================================")

    # 1. ĐỌC DỮ LIỆU RFM
    if not DATA_PATH.exists():
        print(f"[!] Lỗi: Không tìm thấy file dữ liệu tại {DATA_PATH}")
        print("    Vui lòng nhắc Thành viên 5 chạy tiền xử lý trước!")
        return

    print(f"[1/5] Đang đọc dữ liệu từ: {DATA_PATH}")
    df_scaled = pd.read_csv(DATA_PATH)
    features = ['Recency', 'Frequency', 'Monetary']
    X = df_scaled[features].values

    # 2. TÌM SỐ CỤM TỐI ƯU BẰNG BIC VÀ AIC
    print("[2/5] Đang tính BIC và AIC (k từ 2 đến 10)...")
    n_components_range = range(2, 11)
    bic_scores, aic_scores = [], []

    for k in n_components_range:
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    # Lưu biểu đồ BIC/AIC
    plt.figure(figsize=(10, 5))
    plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
    plt.plot(n_components_range, aic_scores, label='AIC', marker='s')
    plt.xlabel('Số lượng cụm (n_components)')
    plt.ylabel('Score')
    plt.title('Lựa chọn số cụm GMM tối ưu bằng BIC & AIC')
    plt.legend()
    plt.grid(True)
    
    bic_fig_path = FIGURES_DIR / "gmm_bic_aic.png"
    plt.savefig(bic_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Đã lưu biểu đồ BIC/AIC tại: {bic_fig_path}")

    # 3. HUẤN LUYỆN GMM TỐI ƯU (ví dụ k=4)
    best_k = 4
    print(f"[3/5] Huấn luyện GMM với k = {best_k}...")
    gmm_best = GaussianMixture(n_components=best_k, random_state=42, covariance_type='full')
    clusters = gmm_best.fit_predict(X)
    probs = gmm_best.predict_proba(X)

    # 4. ĐÁNH GIÁ CHỈ SỐ VÀ VẼ PCA
    print("[4/5] Đánh giá mô hình và vẽ biểu đồ PCA...")
    sil = silhouette_score(X, clusters)
    db = davies_bouldin_score(X, clusters)
    
    print("\n---------------- KẾT QUẢ ĐÁNH GIÁ ----------------")
    print(f" - Silhouette Score    : {sil:.4f}")
    print(f" - Davies-Bouldin Index: {db:.4f}")
    print("--------------------------------------------------\n")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1', alpha=0.7)
    plt.title(f'Trực quan hóa phân cụm GMM (k={best_k}) qua PCA')
    
    cluster_fig_path = FIGURES_DIR / "gmm_clusters_pca.png"
    plt.savefig(cluster_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Đã lưu biểu đồ cụm tại: {cluster_fig_path}")

    # 5. XUẤT MODEL VÀ KẾT QUẢ DỰ ĐOÁN
    print("[5/5] Xuất file mô hình và kết quả phân cụm...")
    model_path = MODELS_DIR / "gmm_model.pkl"
    joblib.dump(gmm_best, model_path)
    print(f" -> Đã lưu mô hình GMM (.pkl) tại: {model_path}")

    df_result = df_scaled.copy()
    df_result['GMM_Cluster'] = clusters
    for i in range(best_k):
        df_result[f'Prob_Cluster_{i}'] = probs[:, i]
    
    output_csv = MODELS_DIR / "gmm_results.csv"
    df_result.to_csv(output_csv, index=False)
    print(f" -> Đã lưu kết quả dữ liệu (.csv) tại: {output_csv}")

    print("\n==================================================")
    print("        HOÀN THÀNH TỐT NHIỆM VỤ THÀNH VIÊN 4!")
    print("==================================================")

if __name__ == "__main__":
    test_or_run_gmm()