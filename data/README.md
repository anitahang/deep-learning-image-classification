# Data Directory

Thư mục này chứa dữ liệu phục vụ nghiên cứu phân loại và nhận diện hình ảnh.

## 1. Cấu trúc thư mục dữ liệu đề xuất
```text
data/
├── raw/                 # Dữ liệu gốc tải về (chưa qua xử lý)
├── processed/           # Dữ liệu sau khi chuẩn hóa kích thước và tăng cường dữ liệu
└── splits/              # Danh sách phân chia train/val/test (.txt hoặc .json)
```

## 2. Nguồn dữ liệu
- Dữ liệu thực nghiệm được thu thập và tiền xử lý theo chuẩn kích thước 224x224 pixels.
- Nhãn phân loại được lưu trữ dưới dạng thư mục theo tên lớp đối tượng.
