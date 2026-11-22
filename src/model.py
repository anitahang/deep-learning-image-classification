import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """Cơ chế chú ý không gian (Spatial Attention Module)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(cat))
        return x * scale

class ImageClassifierWithAttention(nn.Module):
    """
    Mô hình mạng tích chập kết hợp cơ chế chú ý cho bài toán phân loại hình ảnh.
    Nghiên cứu: Ứng dụng học sâu trong bài toán phân loại và nhận diện hình ảnh
    Học viên: Trần Thị Thanh Hằng - ĐH Cần Thơ
    """
    def __init__(self, num_classes=10):
        super(ImageClassifierWithAttention, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.attention = SpatialAttention(kernel_size=7)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat_att = self.attention(feat)
        pooled = self.gap(feat_att)
        flat = torch.flatten(pooled, 1)
        out = self.classifier(flat)
        return out

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    model = ImageClassifierWithAttention(num_classes=10)
    output = model(dummy_input)
    print(f"Mô hình khởi tạo thành công. Shape đầu ra: {output.shape}")
