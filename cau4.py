import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time

def step1(df_name):
    df = pd.read_csv(df_name)
    
    X_raw = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    
    X_mean = np.mean(X_raw, axis=0)
    X_centered = X_raw - X_mean
    
    k_listed = list(range(1, 5))
    
    with open("cau4_output.txt", "w", encoding='utf-8') as f:
        f.write("=== KẾT QUẢ SO SÁNH PCA TỰ CÀI ĐẶT VS SCIKIT-LEARN ===\n\n")
    
    for k in k_listed:
        print(f"\n=== Chiếu dữ liệu xuống không gian {k} chiều ===")
        

        with open("cau4_output.txt", "a", encoding='utf-8') as f:
            f.write(f"=== Chiếu dữ liệu xuống không gian {k} chiều ===\n")
            f.write("So sánh PCA tự cài đặt với PCA của scikit-learn\n")
        
        print("So sánh PCA tự cài đặt với PCA của scikit-learn")
        
        # Đo thời gian chạy tự code PCA
        start_time = time.perf_counter()
        
        cov_matrix = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        custom_eigenvectors = eigenvectors[:, :k]
        custom_eigenvalues = eigenvalues[:k]
        
        X_custom_reduced = np.dot(X_centered, custom_eigenvectors)
        
        end_time = time.perf_counter()
        time_custom = end_time - start_time
        
        # Đo thời gian chạy PCA của scikit-learn
        start_time_sklearn = time.perf_counter()
        pca = PCA(n_components=k)
        X_sklearn_reduced = pca.fit_transform(X_centered)
        sklearn_eigenvectors = pca.components_.T
        sklearn_eigenvalues = pca.explained_variance_
        
        end_time_sklearn = time.perf_counter()
        time_sklearn = end_time_sklearn - start_time_sklearn
        
        # Trường hợp bị ngược dấu
        for i in range(k):
            if np.sign(sklearn_eigenvectors[0, i]) != np.sign(custom_eigenvectors[0, i]):
                sklearn_eigenvectors[:, i] *= -1
                X_sklearn_reduced[:, i] *= -1
                
        # In kết quả so sánh
        np.set_printoptions(precision=6, suppress=True)
        print("1. Tốc độ chạy")
        print(f"Thời gian Numpy tự code : {time_custom:.6f} giây")
        print(f"Thời gian Scikit-learn  : {time_sklearn:.6f} giây\n")
        
        with open("cau4_output.txt", "a", encoding='utf-8') as f:
            f.write("1. Tốc độ chạy\n")
            f.write(f"Thời gian Numpy tự code : {time_custom:.6f} giây\n")
            f.write(f"Thời gian Scikit-learn  : {time_sklearn:.6f} giây\n\n")
            f.write("-" * 50 + "\n\n")
            
        print("\n", '-' * 50, "\n")
        
        print("2. Trị riêng (Eigenvalues) của 2 phương pháp:")
        print(f"Numpy Tự Code (Top {k})  : {custom_eigenvalues}")
        print(f"Scikit-learn (Top {k})   : {sklearn_eigenvalues}")
        error_eigenvalues = np.abs(custom_eigenvalues - sklearn_eigenvalues)
        print(f"=> Sai số lớn nhất (top {k})       : {np.max(error_eigenvalues):.15f}\n")
        
        print('\n', '-' * 50, "\n")
        
        with open("cau4_output.txt", "a", encoding='utf-8') as f:
            f.write("2. Trị riêng (Eigenvalues) của 2 phương pháp:\n")
            f.write(f"Numpy Tự Code (Top {k})  : {custom_eigenvalues}\n")
            f.write(f"Scikit-learn (Top {k})   : {sklearn_eigenvalues}\n")
            f.write(f"=> Sai số lớn nhất (top {k})       : {np.max(error_eigenvalues):.15f}\n\n")
            f.write("-" * 50 + "\n\n")
        
   
        print("3. Dữ liệu sau giảm chiều (Projected Data) của 2 phương pháp:")
        print("Numpy Tự Code (5 mẫu đầu tiên):\n", X_custom_reduced[:5])
        print("\nScikit-learn (5 mẫu đầu tiên):\n", X_sklearn_reduced[:5])
        
        error_reduced = np.abs(X_custom_reduced - X_sklearn_reduced)
        print(f"\n=> Sai số lớn nhất trên toàn bộ 150 mẫu: {np.max(error_reduced):.15f}")


        with open("cau4_output.txt", "a", encoding='utf-8') as f:
            f.write("3. Dữ liệu sau giảm chiều (Projected Data) của 2 phương pháp:\n")
            f.write("Numpy Tự Code (5 mẫu đầu tiên):\n" + str(X_custom_reduced[:5]) + "\n\n")
            f.write("Scikit-learn (5 mẫu đầu tiên):\n" + str(X_sklearn_reduced[:5]) + "\n\n")
            f.write(f"=> Sai số lớn nhất trên toàn bộ 150 mẫu: {np.max(error_reduced):.15f}\n\n")
            f.write("=" * 60 + "\n\n")
            
    return X_centered, eigenvalues, eigenvectors

if __name__ == "__main__":
    df_name = 'Iris.csv'
    X_centered, eigenvalues, eigenvectors = step1(df_name)