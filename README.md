## CSDLDPT
### 1. 🧩 Cài thư viện cần thiết:
```bash
pip install librosa scikit-learn numpy
```

---

### 2. 🛠️ Xây dựng đặc trưng âm thanh cho toàn bộ file `.mp3` trong thư mục dataset:
```bash
python search.py --build
```

Lệnh này sẽ:
- Duyệt qua tất cả file `.mp3` trong thư mục `data/dataset/`
- Trích xuất đặc trưng âm thanh (MFCC, Delta, Chroma, ...)
- Lưu vào file `features.npy` để dùng khi tìm kiếm

---

### 3. 🔍 Tìm file âm thanh tương đồng:
```bash
python search.py --input path/to/your_input.mp3
```

Ví dụ nếu file đầu vào nằm trong thư mục `data/dataset/`:
```bash
python search.py --input data/dataset/test_input.mp3
```

Kết quả sẽ hiển thị tên file `.mp3` trong database có đặc trưng gần giống nhất với file bạn nhập vào.
