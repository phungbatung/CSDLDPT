import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_audio_features
from pymongo import MongoClient

client_url = "mongodb+srv://truongnt:lUH5WK7x5TqjnME0@cluster0.pqgkiks.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

with open("features_range.json", "r") as f:
    min_max_values = json.load(f)

# --- Chuẩn hóa 1 vector feature ---
def normalize_feature(value, min_val, max_val):
    if max_val == min_val:
        return 0
    return 2 * ((value - min_val) / (max_val - min_val)) - 1

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

# --- Tính cosine similarity giữa 2 list vector ---
def average_cosine_similarity(vectors1, vectors2):
    sims = []
    for v1 in vectors1:
        for v2 in vectors2:
            sim = cosine_similarity([v1], [v2])[0][0]
            sims.append(sim)
    return np.mean(sims)

# def cosine_sim(vec1, vec2):
#     """Tính cosine similarity giữa 2 vector numpy."""
#     vec1 = vec1.reshape(1, -1)
#     vec2 = vec2.reshape(1, -1)
#     return cosine_similarity(vec1, vec2)[0][0]

# def average_segment_similarity(df1, df2):
#     """
#     Tính similarity trung bình theo chiến thuật:
#     Với mỗi segment của df1, lấy max similarity với tất cả segment của df2,
#     sau đó trung bình các max similarity này.
    
#     df1, df2: pandas DataFrame chứa các segment feature (không tính cột segment_id, start_time, end_time).
#     """
#     ignore_cols = ['segment_id', 'start_time', 'end_time']
#     features1 = df1.drop(columns=ignore_cols).values
#     features2 = df2.drop(columns=ignore_cols).values

#     max_sims = []
#     for vec1 in features1:
#         sims = []
#         for vec2 in features2:
#             sim = cosine_sim(vec1, vec2)
#             sims.append(sim)
#         if sims:
#             max_sims.append(max(sims))

#     return np.mean(max_sims) if max_sims else 0


def search_similar_audio(file_path, top_k=3):
    # 1. Trích xuất đặc trưng và chuẩn hóa
    segment_result, features_df = extract_audio_features(file_path)
    if features_df.empty:
        print("Không tìm thấy đoạn âm thanh.")
        return

    query_vectors = []
    for _, row in features_df.iterrows():
        norm_vec = normalize_features(row.to_dict())
        query_vectors.append(list(norm_vec.values()))

    # 2. Kết nối MongoDB và truy vấn tất cả đặc trưng
    client = MongoClient(client_url)
    db = client["animal_sounds"]
    collection = db["audio_features"]
    all_docs = list(collection.find())

    # 3. Gom các vector theo filename
    db_features = {}
    for doc in all_docs:
        filename = doc["filename"]
        feature_vec = list(doc["feature"].values())
        db_features.setdefault(filename, []).append(feature_vec)

    # 4. Tính độ tương đồng cosine trung bình
    similarity_scores = []
    for filename, vectors in db_features.items():
        avg_sim = average_cosine_similarity(query_vectors, vectors)
        similarity_scores.append((filename, avg_sim))

    # 5. Sắp xếp và trả kết quả
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarity_scores[:top_k]

    print(f"\nTop {top_k} file giống nhất với {file_path}:")
    for i, (fname, score) in enumerate(top_matches, 1):
        print(f"{i}. {fname} — Cosine Similarity: {score:.4f}")

    return top_matches
