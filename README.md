# 📊 Phân tích thành phần chính (PCA

Repository này chứa toàn bộ bài tập và đồ án nghiên cứu về thuật toán **Principal Component Analysis (PCA)** trong học phần Phương pháp toán cho AI. Dự án đi từ các phép toán nền tảng đến các ứng dụng thực tiễn trong xử lý ảnh và nhận dạng sinh trắc học.

## 📋 Nội dung đồ án

Dự án được chia thành 2 giai đoạn chính:

### 1. Nền tảng và Thực nghiệm (Câu 1 - Câu 5)
* **Câu 1-2:** Triển khai thuật toán PCA cơ bản, tính toán ma trận hiệp phương sai, trị riêng (Eigenvalues) và vector riêng (Eigenvectors).
* **Câu 3-4:** Trực quan hóa dữ liệu và thực hiện giảm chiều trên tập dữ liệu Iris, phân tích sự giữ nguyên thông tin sau khi chiếu.
* **Câu 5: Bài toán Nén ảnh & MSE:** * Thực hiện nén ảnh chân dung từ $N$ chiều xuống $k$ chiều.
    * Phân tích sai số bình phương trung bình (**MSE**) giữa ảnh gốc và ảnh tái tạo.
    * Đánh giá hiệu quả lọc nhiễu (Denoising) thông qua việc loại bỏ các thành phần chính có giá trị riêng thấp.

### 2. Ứng dụng thực tế (Câu Bonus - Eigenfaces)
* **Nhận dạng khuôn mặt:** Xây dựng hệ thống nhận diện trên tập dữ liệu **AT&T Face Database**.
* **Thủ thuật Turk & Pentland:** Triển khai phương pháp tính toán tối ưu ma trận hiệp phương sai để tránh lỗi tràn bộ nhớ (Out-of-memory).
* **Phân loại K-NN:** Tinh chỉnh tham số $k$ trong thuật toán Láng giềng gần nhất để tối ưu hóa độ chính xác (Accuracy đạt 95% với $k_{NN}=1$).

## 🛠 Công nghệ & Thư viện
* **Ngôn ngữ:** Python 3.10.x
* **Thư viện chính:** * `NumPy`: Xử lý ma trận và đại số tuyến tính (không dùng thư viện nén sẵn).
    * `Matplotlib`: Trực quan hóa dữ liệu và hiển thị ảnh.
* **Báo cáo:** Soạn thảo bằng LaTeX chuyên nghiệp.

## 📈 Kết quả nổi bật
* **Lọc nhiễu:** Chứng minh được PCA hoạt động như một bộ lọc thông thấp, giúp làm mịn ảnh bằng cách loại bỏ nhiễu hạt ở các chiều dữ liệu cuối.
* **Nén dữ liệu:** Đạt tỷ lệ nén cao nhưng vẫn giữ được các đặc trưng nhận dạng cốt lõi của khuôn mặt.
* **So sánh PCA vs LDA:** Phân tích sự khác biệt giữa phương pháp không giám sát (PCA - tối đa phương sai) và có giám sát (LDA - tối đa khoảng cách giữa các lớp).

## 📂 Cấu trúc thư mục
```text
├── eigenfaces_dataset/ # Dữ liệu ảnh AT&T (40 người)
├── code_bai_tap.py     # Giải quyết từ Câu 1 đến Câu 5
├── cau_bonus_face.py   # Code nhận dạng khuôn mặt Eigenfaces
├── Bao_cao_PCA.pdf     # Bản báo cáo chi tiết cuối cùng
└── main.tex            # Mã nguồn LaTeX của báo cáo
