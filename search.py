import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_audio_features

def cosine_sim(vec1, vec2):
    """Tính cosine similarity giữa 2 vector numpy."""
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def average_segment_similarity(df1, df2):
    """
    Tính similarity trung bình theo chiến thuật:
    Với mỗi segment của df1, lấy max similarity với tất cả segment của df2,
    sau đó trung bình các max similarity này.
    
    df1, df2: pandas DataFrame chứa các segment feature (không tính cột segment_id, start_time, end_time).
    """
    ignore_cols = ['segment_id', 'start_time', 'end_time']
    features1 = df1.drop(columns=ignore_cols).values
    features2 = df2.drop(columns=ignore_cols).values

    max_sims = []
    for vec1 in features1:
        sims = []
        for vec2 in features2:
            sim = cosine_sim(vec1, vec2)
            sims.append(sim)
        if sims:
            max_sims.append(max(sims))

    return np.mean(max_sims) if max_sims else 0


def search_similar_audio(input_path, top_k=3):
    """
    Tìm kiếm các file âm thanh tương tự nhất trong thư mục dựa trên đặc trưng âm thanh.
    """
    feature_dir = 'data/feature_output'
    
    # Đọc đặc trưng file input
    input_features_df = extract_audio_features(input_path)[1]

    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('_features.csv')]
    similarity_scores = []

    for feat_file in feature_files:
        file_path = os.path.join(feature_dir, feat_file)
        # print(f"Đang so sánh với {file_path}...")
        # Đọc dataframe đặc trưng
        df = pd.read_csv(file_path)

        # Tính similarity trung bình giữa input và file này
        try:
            sim = average_segment_similarity(input_features_df, df)
        except Exception as e:
            print(f"⚠ Lỗi khi so sánh với {file_path}: {e}")
            sim = 0
        similarity_scores.append((feat_file, sim))

    # Sắp xếp theo similarity giảm dần
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # In top_k file
    print(f"Top {top_k} file giống nhất với {input_path}:")
    for i, (fname, score) in enumerate(similarity_scores[:top_k], 1):
        print(f"{i}. {fname} - Similarity: {score:.4f}")

    # Trả về danh sách top_k file và điểm similarity
    return similarity_scores[:top_k]
