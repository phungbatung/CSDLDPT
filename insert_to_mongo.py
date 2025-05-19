import os
import json
import librosa
import pandas as pd
from pymongo import MongoClient
from feature_extractor import extract_audio_features  # Đảm bảo hàm của bạn ở đây hoặc cùng file

client_url = "mongodb+srv://truongnt:lUH5WK7x5TqjnME0@cluster0.pqgkiks.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# --- Giá trị Min-Max để chuẩn hóa ---
with open("features_range.json", "r") as f:
    min_max_values = json.load(f)

def normalize_feature(value, min_val, max_val):
    if max_val == min_val:
        return 0  # tránh chia 0
    return 2 * ((value - min_val) / (max_val - min_val)) - 1  # Chuẩn hóa về [-1, 1]

def normalize_features(row):
    normalized = {}
    for key, value in row.items():
        if key.startswith("mfcc_"):
            mfcc_idx = key
            mfcc_min = min_max_values["mfcc"][mfcc_idx]["min"]
            mfcc_max = min_max_values["mfcc"][mfcc_idx]["max"]
            normalized[key] = normalize_feature(value, mfcc_min, mfcc_max)
        elif key in min_max_values:
            min_val = min_max_values[key]["min"]
            max_val = min_max_values[key]["max"]
            normalized[key] = normalize_feature(value, min_val, max_val)
    return normalized

def process_audio_folder(audio_folder):
    client = MongoClient(client_url)
    db = client["animal_sounds"]
    collection = db["audio_features"]

    for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".mp3") or file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Đang xử lý file: {file_path}")

                try:
                    segment_result, features_df = extract_audio_features(file_path)
                    if features_df.empty:
                        continue

                    # Duyệt qua từng dòng (segment) trong DataFrame
                    for _, row in features_df.iterrows():
                        normalized_feature = normalize_features(row.to_dict())
                        doc = {
                            "filename": file,
                            "feature": normalized_feature
                        }
                        collection.insert_one(doc)
                        print(f"Đã lưu đặc trưng cho segment trong file: {file}")
                except Exception as e:
                    print(f"Lỗi khi xử lý {file_path}: {e}")

if __name__ == "__main__":
    audio_folder = r"data\dataset"
    process_audio_folder(audio_folder)


