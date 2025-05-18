import os
import sys
import argparse
from feature_extractor import extract_audio_features, visualize_segments
from search import search_similar_audio  # import hàm search bạn đã có

def process_audio_file(file_path, output_dir, window_size, overlap, k, threshold):
    print(f"=== Đang xử lý file: {file_path} ===")
    
    # Trích xuất segment & đặc trưng
    segment_result, features_df = extract_audio_features(file_path)

    # Lưu đặc trưng ra CSV (tên giống file gốc)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_csv = os.path.join(output_dir, f"{file_name}_features.csv")
    features_df.to_csv(output_csv, index=False)
    print(f">>> Đã lưu đặc trưng vào {output_csv}")

    # Vẽ waveform với segment
    # visualize_segments(file_path, segment_result, window_size, overlap)

def build_features_for_all(input_dir, output_dir, window_size, overlap, k, threshold):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.mp3') or file_name.lower().endswith('.wav'):
            file_path = os.path.join(input_dir, file_name)
            process_audio_file(file_path, output_dir, window_size, overlap, k, threshold)

def main():
    parser = argparse.ArgumentParser(description="Audio feature extractor and search tool")
    parser.add_argument('--build', action='store_true', help='Extract features for all files in dataset')
    parser.add_argument('--search', type=str, help='Search similar audios for given file path')

    args = parser.parse_args()

    # Các tham số xử lý (bạn có thể tùy chỉnh)
    window_size = 0.085
    overlap = 0.3
    k = 2
    threshold = 0.3

    input_dir = 'data/dataset'
    output_dir = 'data/feature_output'

    if args.build:
        print("Bắt đầu trích xuất đặc trưng tất cả file trong thư mục dataset...")
        build_features_for_all(input_dir, output_dir, window_size, overlap, k, threshold)
    elif args.search:
        file_to_search = args.search
        print(f"Tìm kiếm các file tương tự với file: {file_to_search}")
        search_similar_audio(file_to_search, top_k=3)
    else:
        print("Bạn cần truyền tham số --build hoặc --search filepath")
        parser.print_help()

if __name__ == "__main__":
    main()
