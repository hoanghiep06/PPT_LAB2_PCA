import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

def load_att_faces(dataset_path):
    X_train, y_train = [], []
    X_test, y_test = [], []
    original_shape = None
    
    print("Đang đọc dữ liệu trong ", dataset_path)
    
    # Dataset này có 40 người
    for i in range(1, 41):
        subject_folder = os.path.join(dataset_path, f"s{i}")
        
        # Mỗi người có 10 ảnh
        for j in range(1, 11):
            img_path = os.path.join(subject_folder, f"{j}.pgm")
            
            img = plt.imread(img_path)
            if original_shape is None:
                original_shape = img.shape
            
            # Chuẩn hóa pixel từ 0 - 255 về 0 - 1
            if img.max() > 1.0:
                img = img / 255.0
            
            # Flatten ảnh từ 2D sang 1D
            img_flat = img.flatten()
            
            # Chia test, train: 8 tấm đầu để train, 2 tấm cuối để test
            if j <= 8:
                X_train.append(img_flat)
                y_train.append(f"Người {i}")
            else:
                X_test.append(img_flat)
                y_test.append(f"Người {i}")
                
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), original_shape

if __name__ == "__main__":
    dataset_path = "./eigenfaces_dataset"
    
    # Bước 1: load data
    X_train, y_train, X_test, y_test, shape = load_att_faces(dataset_path)
    
    print(f"Kích thước 1 bức ảnh gốc: {shape}")
    print(f"Tập Train (X_train) : {X_train.shape[0]} ảnh, ma trận kích thước {X_train.shape}")
    print(f"Tập Test  (X_test)  : {X_test.shape[0]} ảnh, ma trận kích thước {X_test.shape}")
    
    # Trung bình khuôn mặt và chuẩn hóa data
    mean_faces = np.mean(X_train, axis=0)
    X_centered = X_train - mean_faces   
    
    # Tính ma trận L = X. X(T) (320x320) theo thuật toán Turk & Pentland (1991)thay vì tính Cov_matrix = X(T). X (10304x10304) sẽ rất lâu
    L = np.dot(X_centered, X_centered.T) / (X_train.shape[0] - 1)
    
    # Trị riêng và vector riêng
    eigenvalues, eigenvectors_L = np.linalg.eigh(L)
    
    # Tính vector riêng thực sự của ảnh thông qua X_centered(T).eigenvectors_L 
    eigenvectors = np.dot(X_centered.T, eigenvectors_L)
    
    # Chuẩn hóa độ dài vector riêng về vector đơn vị
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    
    # Sort trị riêng
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    
    # Chọn k
    k = 50
    U_k = eigenvectors[:, :k]   
    print(f"Trích xuất thành công {k} mặt đặc trưng")
    
    # Chiếu dữ liệu
    train_proj = np.dot(X_centered, U_k)
    print(f"Kích thước X_train sau khi nén: {train_proj.shape}")
    
    # In 
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # Vẽ khuôn mặt trung bình
    axes[0].imshow(mean_faces.reshape(shape), cmap='gray')
    axes[0].set_title("Mean Face")
    axes[0].axis('off')
    
    # Top 3 eigenfaces max
    for i in range(3):
        eigenface_img = U_k[:, i].reshape(shape) 
        axes[i+1].imshow(eigenface_img, cmap='bone')
        axes[i+1].set_title(f"Eigenface {i+1}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('cau_bonus.png', dpi=300)
    plt.show()
    
    
    # Testing
    print("\n Bước 3: Kiểm thử")
    print('\n', '-' * 50, '\n')
    
    correct_pred = 0
    total_images_test = len(X_test)
    kNN_lst = list(range(1, 11))
    for kNN in kNN_lst:
        correct_pred = 0
        for i in range(total_images_test):
            # Dùng mean tập train
            test_centered = X_test[i] - mean_faces 
            
            # Chiếu ảnh test lên không gian eigenfaces (từ 10304D -> 50D)
            test_proj = np.dot(test_centered, U_k) # (320, 50)
            
            # Tính khoảng cách Euclid từ ảnh test tới 320 ảnh TRAIN
            distances = np.linalg.norm(train_proj - test_proj, axis=1)
            
            # Dùng thuật toán kNN: tìm ảnh train có dist gần nhất, không quan tâm trọng số gần xa chỉ lấy k tấm gần nhất đem so
            best_match_idx = np.argpartition(distances, kNN)[:kNN]
            predicted_label_lst = y_train[best_match_idx]
            most_common = Counter(predicted_label_lst).most_common(1)
            predicted_label = most_common[0][0]
            
            actual_label = y_test[i] 
            
            if predicted_label == actual_label:
                correct_pred += 1
            
            # Metric
        accuracy = (correct_pred / total_images_test) * 100
        print(f"KNN = {kNN}")
        print(f"Tổng số ảnh Test : {total_images_test}")
        print(f"Dự đoán đúng     : {correct_pred}")
        print(f"Độ chính xác     : {accuracy:.4f}%\n")
        
        print("="*50)