from feature_extractor import extract_audio_features
import os
import pandas as pd
import json

# Đường dẫn tới thư mục chứa audio
audio_folder = r"data\dataset"

# Tham số cho hàm trích xuất đặc trưng
window_size = 0.1     # Giây
overlap = 0.3         # Phần trăm chồng lấn
k = 1                 # Số cửa sổ so sánh
threshold = 0.3       # Ngưỡng spectral centroid

# Khởi tạo list để lưu toàn bộ các đặc trưng
all_features = []

# Duyệt qua các file trong thư mục
for filename in os.listdir(audio_folder):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        file_path = os.path.join(audio_folder, filename)
        print(f"Đang xử lý: {file_path}")

        try:
            segment_result, features_df = extract_audio_features(
                file_path,
                window_size,
                overlap,
                k,
                threshold
            )

            print("Số segment:", len(features_df))

            all_features.append(features_df)

        except Exception as e:
            print(f"Lỗi khi xử lý {file_path}: {e}")

# Gộp tất cả các DataFrame lại thành một DataFrame lớn
if all_features:
    combined_df = pd.concat(all_features, ignore_index=True)

    # Tính mean và std cho từng cột đặc trưng
    feature_stats = {}
    for column in combined_df.columns:
        mean = combined_df[column].mean()
        std = combined_df[column].std()
        feature_stats[column] = {
            "mean": float(mean),
            "std": float(std)
        }

    # Lưu ra file JSON
    output_path = "features_stats_zscore.json"
    with open(output_path, "w") as f:
        json.dump(feature_stats, f, indent=2)

    print(f"\n✅ Đã lưu mean/std (cho chuẩn hóa Z-score) vào: {output_path}")
else:
    print("❌ Không có file nào được xử lý.")
