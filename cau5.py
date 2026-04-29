import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_image_compression(image_path):
    # Cố định seed
    print(" BÀI TOÁN NÉN VÀ TÁI TẠO ẢNH BẰNG PCA")
    print("-" * 50)
    
    img_raw = plt.imread(image_path)
    
    # Chuẩn hóa từ 0 - 255 về 0 - 1
    if img_raw.max() > 1.0:
        img_raw = img_raw / 255.0
        
    # Đổi RGB sang GrayScale 
    if len(img_raw.shape) == 3:
        if img_raw.shape[2] == 4: 
            # XỬ LÝ ẢNH 4 KÊNH (RGBA)
            alpha = img_raw[..., 3:]   
            rgb = img_raw[..., :3]     
            bg_color = np.array([1.0, 1.0, 1.0]) 
            
            # Công thức Alpha Compositing: Pixel_mới = RGB * Alpha + Nền * (1 - Alpha)
            img_rgb = rgb * alpha + bg_color * (1.0 - alpha)
        else:
            # ẢNH 3 KÊNH (RGB)
            img_rgb = img_raw[..., :3]
            
        # Chuyển RGB sang Grayscale
        X_gray = np.dot(img_rgb, [0.2989, 0.5870, 0.1140])
    else:
        # ẢNH 1 KÊNH (Đã là Grayscale sẵn)
        X_gray = img_raw
    
    # Xem ảnh
    # plt.imshow(X_gray, cmap='gray')
    # plt.show()

    # Chuẩn hóa dữ liệu 
    X = np.array(X_gray, dtype=np.float64)
    mean_vector = np.mean(X, axis=0)
    X_centered = X - mean_vector
    
    print(f"Kích thước ma trận ảnh gốc: {X.shape}")
    
    # Tính toán PCA
    cov_matrix = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Các giá trị k thử nghiệm
    k_list = [2, 5, 10, 20, 50, 100, 200]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    # Ảnh gốc ô 1
    axes[0].imshow(X, cmap='gray')
    axes[0].set_title("Ảnh gốc ")
    axes[0].axis('off')
    
    for i, k in enumerate(k_list):
        U_k = eigenvectors[:, :k]
        
        # Giảm chiều (compress)
        X_reduced = np.dot(X_centered, U_k)    
            
        # Tái tạo lại ảnh gốc từ ảnh bị giảm chiều
        X_reconstructed = np.dot(X_reduced, U_k.T) + mean_vector
        
        # Đánh giá dựa trên lỗi tái tạo (MSE)
        mse = np.mean((X_centered - X_reconstructed) ** 2)
        print(f"- Với k = {k:3d} | Lỗi tái tạo (MSE): {mse:.4f}")
        
        # Vẽ ảnh lên
        axes[i + 1].imshow(X_reconstructed, cmap='gray')
        axes[i + 1].set_title(f"Tái tạo k={k}\nMSE: {mse:.4f}")
        axes[i + 1].axis('off')
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('cau5.png', dpi=300)

    plt.show()
    
if __name__ == "__main__":
    image_path = './lfw_deepfunneled/Abdel_Aziz_Al_Hakim_0001.jpg'
    run_image_compression(image_path)
    