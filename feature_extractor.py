import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_audio_features(file_path, window_size=0.1, overlap=0.3, k=1, threshold=0.3, rms_threshold=0.055):
    """
    Trích xuất đặc trưng âm thanh theo segment dựa trên spectral_centroid.
    
    Parameters:
        file_path (str): Đường dẫn file âm thanh.
        window_size (float): Kích thước cửa sổ (giây).
        overlap (float): Phần trăm chồng lấn giữa các cửa sổ (0~1).
        k (int): So sánh window i với window (i-k).
        threshold (float): Ngưỡng chênh lệch spectral_centroid để tách đoạn.
        rms_threshold (float): Ngưỡng năng lượng RMS để lọc segment yếu.
    
    Returns:
        segment_result (list): Danh sách các segment (start_frame, end_frame).
        features_df (pd.DataFrame): Bảng đặc trưng của các segment.
    """
        # Load file âm thanh
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Tính số mẫu cho mỗi cửa sổ và bước nhảy
    window_length = int(window_size * sr)
    hop_length = int(window_length * (1 - overlap))

    # Tính RMS energy cho toàn bộ file theo từng window
    rms_energy = librosa.feature.rms(y=y, frame_length=window_length, hop_length=hop_length)[0]
    num_windows = len(rms_energy)

    #Test rms_threshold
    # rms_threshold = np.mean(rms_energy)*0.5
    # print(f"Ngưỡng RMS: {rms_threshold:.4f}")
    
    # Tách segment dựa trên chênh lệch RMS energy
    segment_result = []
    start_segment = 0

    for i in range(k, num_windows):
        max_val = max(rms_energy[i], rms_energy[i - k])
        diff = abs(rms_energy[i] - rms_energy[i - k])
        if max_val != 0:  # Tránh chia 0
            percent_diff = diff / max_val
        else:
            percent_diff = 0
        # print(f"Window {i}: {rms_energy[i]} vs {rms_energy[i - k]} -> Percent diff: {percent_diff:.4f}")
        # Nếu chênh lệch lớn hơn ngưỡng, đánh dấu segment
        # if percent_diff > threshold and diff > 0.05:  # diff > 0.01 là ngưỡng RMS tuyệt đối
        # print(f"Window {i}: {rms_energy[i]} ")
        if rms_energy[i] < 0.04:
            segment_result.append((start_segment, i))
            start_segment = i - k


    # Đảm bảo segment cuối cùng được thêm vào
    if start_segment < num_windows - 1:
        segment_result.append((start_segment, num_windows - 1))

    # Trích xuất đặc trưng cho từng segment
    feature_list = []
    filtered_segment_result = []
    for seg_id, (start, end) in enumerate(segment_result):
        start_sample = start * hop_length
        end_sample = min(end * hop_length + window_length, len(y))
        y_seg = y[start_sample:end_sample]

        # Tính năng lượng RMS trung bình cho segment
        rms = np.mean(librosa.feature.rms(y=y_seg))
        # Nếu RMS nhỏ hơn ngưỡng, bỏ qua segment này
        # print(f"Segment {seg_id + 1}: RMS = {rms:.4f}, Threshold = {rms_threshold}")
        if rms < rms_threshold:
            continue
        else:
            print(f"Segment {seg_id}: RMS = {rms:.4f} - Đã thêm vào danh sách")
            filtered_segment_result.append((start, end))

        # Trích xuất các đặc trưng
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y_seg))
        spectral_centroid_seg = np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_seg, sr=sr))
        # spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y_seg, sr=sr))
        mfcc = np.mean(librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13), axis=1)

        feature_row = {
            'segment_id': seg_id + 1,
            'start_time': start * window_size * (1 - overlap),
            'end_time': end * window_size * (1 - overlap),
            'rms_energy': rms,  # Bổ sung giá trị RMS vào kết quả
            'zero_crossing_rate': zero_crossing_rate,
            'spectral_centroid': spectral_centroid_seg,
            'spectral_bandwidth': spectral_bandwidth
            # 'spectral_contrast': spectral_contrast
        }

        # Thêm MFCCs
        for idx, mfcc_val in enumerate(mfcc):
            feature_row[f'mfcc_{idx+1}'] = mfcc_val
        feature_list.append(feature_row)
    if len(feature_list) == 0:
        print("Không tìm thấy segment nào thỏa mãn điều kiện.")
        #trích xuất đặc trưng cho cả file
        rms = np.mean(librosa.feature.rms(y=y))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spectral_centroid_seg = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        feature_row = {
            'segment_id': 0,
            'start_time': 0,
            'end_time': total_duration,
            'rms_energy': rms,  # Bổ sung giá trị RMS vào kết quả
            'zero_crossing_rate': zero_crossing_rate,
            'spectral_centroid': spectral_centroid_seg,
            'spectral_bandwidth': spectral_bandwidth
        }
        # Thêm MFCCs
        for idx, mfcc_val in enumerate(mfcc):
            feature_row[f'mfcc_{idx+1}'] = mfcc_val
        feature_list.append(feature_row)
        # Tạo DataFrame từ danh sách đặc trưng
        filtered_segment_result = [(0, num_windows - 1)]
    features_df = pd.DataFrame(feature_list)

    return filtered_segment_result, features_df


# Vẽ các segment trên waveform
def visualize_segments(file_path, segment_result, window_size=2.0, overlap=0.5):
    # Load file âm thanh
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Vẽ waveform
    plt.figure(figsize=(14, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title("Waveform with Segments")

    # Vẽ line + đánh số trên từng line
    for idx, (start_idx, end_idx) in enumerate(segment_result):
        start_time = start_idx * window_size * (1 - overlap)
        end_time = end_idx * window_size * (1 - overlap)

        # Vạch start (xanh lá)
        plt.axvline(x=start_time, color='green', linestyle='--', alpha=0.7)
        plt.text(start_time, 1.05, f'S{idx+1}', ha='center', va='bottom', fontsize=9, color='green', transform=plt.gca().get_xaxis_transform())

        # Vạch end (đỏ)
        plt.axvline(x=end_time, color='red', linestyle='--', alpha=0.7)
        plt.text(end_time, 1.05, f'S{idx+1}', ha='center', va='bottom', fontsize=9, color='red', transform=plt.gca().get_xaxis_transform())

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
# --- Chạy thử ---
def test_extract_features():
    file_path = r'data\dataset\Rooster-crowing-3-times-sound-effect.mp3'
    # Đường dẫn tới file âm thanh
    window_size = 0.1  # Kích thước cửa sổ (giây)
    overlap = 0.3  # Tỉ lệ chồng lấn giữa các cửa sổ
    k = 1  # Số lượng cửa sổ để so sánh
    threshold = 0.3  # Ngưỡng chênh lệch spectral centroid để tách đoạn
    segment_result, features_df = extract_audio_features(file_path, window_size, overlap, k, threshold)

    # Xem kết quả
    print("Tổng số segment: " + str(len(features_df)))
    print("Danh sách segment (bắt đầu, kết thúc):")
    print(segment_result)
    visualize_segments(file_path, segment_result,  window_size, overlap)

# test_extract_features()
