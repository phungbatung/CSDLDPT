# Hệ thống tìm kiếm âm thanh động vật

Đây là hệ thống tìm kiếm âm thanh động vật dựa trên việc trích xuất đặc trưng từng đoạn âm thanh và so sánh độ tương đồng bằng độ đo cosine.

## 1. Cài đặt thư viện

Trước tiên, hãy cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install librosa scikit-learn numpy matplotlib pandas flask
```

## 2. Xây dựng đặc trưng cho các file trong cơ sở dữ liệu

Trích xuất đặc trưng và lưu trữ vào hệ thống bằng lệnh sau:

```bash
python handler.py --build
```

## 3. Chạy hệ thống

Sau khi xây dựng xong cơ sở dữ liệu đặc trưng, bạn có thể khởi động hệ thống để thử nghiệm:

```bash
python main.py
```

Hệ thống sẽ tạo một server Flask đơn giản để nhận âm thanh đầu vào, xử lý và trả về các kết quả âm thanh tương đồng nhất từ cơ sở dữ liệu.
