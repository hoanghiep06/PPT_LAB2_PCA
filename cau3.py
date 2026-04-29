import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def step1(df_name): # Đọc dữ liệu
    df = pd.read_csv(df_name)
    
    # print("Thông tin về dataset:")
    # print(df.info())
    
    # print('\n', "-" * 50, "\n")
    # print("5 dòng đầu tiên của dataset:")
    # print(df.head())
    
    # # Các giá trị thống kê cơ bản
    # print('\n', "-" * 50, "\n")
    # print("Các giá trị thống kê cơ bản:")
    # print(df.describe())
    
    # # Kiểm tra các giá trị có của mỗi cột
    # print('\n', "-" * 50, "\n")
    # print("Các giá trị có của mỗi cột:")
    # for column in df.columns:
    #     if column == 'Id':  # Bỏ qua cột Id
    #         continue
    #     unique_values = df[column].unique()
    #     print(f"{column}: {unique_values}")
    
    print('\n', "-" * 50, "\n")
    # Lấy ma trận giá trị từ cột (SepalLengthCm, SepalWidthCm) làm dữ liệu 
    X_raw = df[['SepalLengthCm', 'SepalWidthCm']].values
    
    # Chuẩn hóa dữ liệu (trừ mean)
    mean_vector = np.mean(X_raw, axis=0)
    X_centered = X_raw - mean_vector
    
    # Vẽ đồ thị dữ liệu gốc
    plt.figure(figsize=(8, 6))
    plt.scatter(X_centered[:, 0], X_centered[:, 1], c='blue', edgecolor='k', s=100)
    
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    
    plt.title('Dữ liệu sau khi chuẩn hóa (trừ mean)')
    plt.xlabel('SepalLengthCm (trừ mean)')
    plt.ylabel('SepalWidthCm (trừ mean)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.axis('equal')
    # Hình ảnh về 
    plt.savefig('cau3_y1.png', dpi=300, bbox_inches='tight')
    
    # plt.show()
    return X_centered
def step2(X_centered):
    # Tính PCA
    cov_matrix = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Vẽ đò thị
    plt.figure(figsize=(8, 6))
    plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5, color='blue', edgecolors='k', label='Dữ liệu gốc (Mean-centered)')
    
    colors = ['red', 'green']
    labels = ['PC1 (Max Variance)', 'PC2 (Orthogonal)']
    
    for i in range(2):
        vec = eigenvectors[:, i] 
        length = np.sqrt(eigenvalues[i]) * 2  
        
        plt.quiver(0, 0, vec[0] * length, vec[1] * length, 
                   angles='xy', scale_units='xy', scale=1, 
                   color=colors[i], label=labels[i], width=0.008)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.title("Trực quan hóa Dữ liệu gốc và Các thành phần chính (PCA)")
    plt.xlabel("Sepal Length (centered)")
    plt.ylabel("Sepal Width (centered)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    
    # Hình ảnh về Dữ liệu gốc và các thành phần chính
    plt.savefig('cau3_y2.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return eigenvalues, eigenvectors

def step3(X_centered, eigenvectors):
    k = 1 # Giảm xuống 1D
    
    # Chiếu dữ liệu xuống không gian mới
    projection_matrix = eigenvectors[:, :k]
    X_reduced = np.dot(X_centered, projection_matrix)
    X_projected_2D = np.outer(X_reduced, projection_matrix) # Chiếu đầy đủ để so sánh
    
    print("Dữ liệu sau khi chiếu xuống PC1:")
    print(X_reduced)

    print('\n', "-" * 50, "\n")
    
    plt.figure(figsize=(10, 3))
    plt.scatter(X_reduced, np.zeros_like(X_reduced), alpha=0.6, color='purple', edgecolors='k')
    
    plt.axhline(0, color='black', linewidth=1.5)
    plt.title("Dữ liệu sau khi giảm chiều xuống 1D")
    plt.xlabel("Principal Component 1")
    
    plt.yticks([]) 
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    # Hình ảnh về Dữ liệu sau khi chiếu xuống 1D
    plt.savefig('cau3_y3.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("Dữ liệu sau khi chiếu xuống không gian mới (2D) để so sánh:")
    print(X_projected_2D)
    
    plt.figure(figsize=(8, 6))
    
    for i in range(X_centered.shape[0]):
        plt.plot([X_centered[i, 0], X_projected_2D[i, 0]], 
                 [X_centered[i, 1], X_projected_2D[i, 1]], 
                 color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    scale = 2.5
    plt.plot([-projection_matrix[0] * scale, projection_matrix[0] * scale],
             [-projection_matrix[1] * scale, projection_matrix[1] * scale],
             color='red', label='Truc PC1 ', linewidth=2, zorder=2)
    
    # Vẽ dữ liệu gốc
    plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6, color='blue', edgecolors='k', label='Dữ liệu gốc (Mean-centered)', zorder=3)
    
    # Vẽ dữ liệu sau khi chiếu
    plt.scatter(X_projected_2D[:, 0], X_projected_2D[:, 1], alpha=0.6, color='purple', edgecolors='k', label='Dữ liệu sau khi chiếu (PC1)', zorder=4)   
    
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linewidth=1, alpha=0.5)
    
    plt.title("So sánh Dữ liệu gốc và Dữ liệu sau khi chiếu xuống PC1")
    plt.xlabel("Sepal Length (centered)")
    plt.ylabel("Sepal Width (centered)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    
    # Minh họa không gian dữ liệu gốc và kết quả thu được sau khi thực hiện phép chiếu lên thành phần chính thứ nhất
    plt.savefig('cau3_y3_sosanh.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
if __name__ == "__main__":
    
    X_centered = step1('Iris.csv')
    eigenvalues, eigenvectors = step2(X_centered)
    step3(X_centered, eigenvectors)