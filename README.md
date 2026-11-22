# Ung dung hoc sau trong bai toan phan loai va nhan dien hinh anh

## 1. Gioi thieu
Du an nghien cuu mo hinh hoc sau ket hop mang tich chap va co che chu y nham toi uu hoa do chinh xac va do tre trong bai toan phan loai hinh anh thoi gian thuc.

## 2. Cau truc thu muc
- `data/`: Du lieu mau va huong dan chuan bi dataset.
- `src/`: Ma nguon mo hinh, script huan luyen va danh gia thuc nghiem.
- `docs/`: So do kien truc mo hinh va tai lieu mo ta.
- `results/`: Bieu do ket qua, ban do dac trung va chi so danh gia.

## 3. Ket qua thuc nghiem
- Do chinh xac (Accuracy): 95.64%
- F1-Score: 95.41%
- Do tre suy luan: 14.1 ms/khung hinh tren GPU NVIDIA RTX 3080 Ti.

## 4. Cai dat
```bash
pip install -r requirements.txt
python src/evaluate.py
```

## 5. Ban quyen va Trich dan
- Du an su dung giay phep MIT License (xem file LICENSE).
- Huong dan trich dan duoc mo ta tai file CITATION.cff.
