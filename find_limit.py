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

# Khởi tạo dictionary chứa min/max cho các đặc trưng
features_range = {
    "rms_energy": {"min": 999999999, "max": -999999999},
    "zero_crossing_rate": {"min": 999999999, "max": -999999999},
    "spectral_centroid": {"min": 999999999, "max": -999999999},
    "spectral_bandwidth": {"min": 999999999, "max": -999999999},
    "mfcc": {f"mfcc_{i}": {"min": 999999999, "max": -999999999} for i in range(1, 14)}
}

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

            # Cập nhật min/max cho từng đặc trưng
            for _, row in features_df.iterrows():
                features_range["rms_energy"]["min"] = min(features_range["rms_energy"]["min"], row["rms_energy"])
                features_range["rms_energy"]["max"] = max(features_range["rms_energy"]["max"], row["rms_energy"])

                features_range["zero_crossing_rate"]["min"] = min(features_range["zero_crossing_rate"]["min"], row["zero_crossing_rate"])
                features_range["zero_crossing_rate"]["max"] = max(features_range["zero_crossing_rate"]["max"], row["zero_crossing_rate"])

                features_range["spectral_centroid"]["min"] = min(features_range["spectral_centroid"]["min"], row["spectral_centroid"])
                features_range["spectral_centroid"]["max"] = max(features_range["spectral_centroid"]["max"], row["spectral_centroid"])

                features_range["spectral_bandwidth"]["min"] = min(features_range["spectral_bandwidth"]["min"], row["spectral_bandwidth"])
                features_range["spectral_bandwidth"]["max"] = max(features_range["spectral_bandwidth"]["max"], row["spectral_bandwidth"])

                for i in range(1, 14):
                    key = f"mfcc_{i}"
                    features_range["mfcc"][key]["min"] = min(features_range["mfcc"][key]["min"], row[key])
                    features_range["mfcc"][key]["max"] = max(features_range["mfcc"][key]["max"], row[key])

        except Exception as e:
            print(f"Lỗi khi xử lý {file_path}: {e}")

# Lưu kết quả ra file JSON
output_path = "features_range.json"
with open(output_path, "w") as f:
    json.dump(features_range, f, indent=2)

print(f"\n✅ Đã lưu min/max đặc trưng vào: {output_path}")
