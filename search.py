import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_features

DATASET_DIR = "data/dataset"
FEATURES_PATH = "data/features.npy"
FILES_PATH = "data/files.npy"

def build_feature_database():
    features = []
    file_names = []

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".mp3"):
            path = os.path.join(DATASET_DIR, file)
            try:
                vec = extract_features(path)
                features.append(vec)
                file_names.append(file)
                print(f"✅ Processed: {file}")
            except Exception as e:
                print(f"❌ Error with {file}: {e}")

    np.save(FEATURES_PATH, np.array(features))
    np.save(FILES_PATH, np.array(file_names))
    print("✅ Feature database saved!")

def search_similar_audio(input_path, top_k=3):
    db_features = np.load(FEATURES_PATH)
    file_names = np.load(FILES_PATH)

    input_vec = extract_features(input_path).reshape(1, -1)
    sims = cosine_similarity(input_vec, db_features)[0]
    
    top_indices = sims.argsort()[-top_k:][::-1]
    results = [(file_names[i], sims[i]) for i in top_indices]

    print(f"\n🔍 Top {top_k} similar audio files:")
    for name, score in results:
        print(f"- {name}: similarity = {score:.4f}")

# ------ DÙNG ------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Xây dựng lại cơ sở dữ liệu đặc trưng")
    parser.add_argument("--input", type=str, help="Đường dẫn tới file .mp3 cần tìm tương đồng")
    args = parser.parse_args()

    if args.build:
        build_feature_database()

    if args.input:
        search_similar_audio(args.input)