import soundfile as sf
import os

# Thư mục chứa các file âm thanh
audio_folder = "data\dataset"

# Lặp qua từng file trong thư mục
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav") or filename.endswith(".flac") or filename.endswith(".mp3"):
        filepath = os.path.join(audio_folder, filename)
        try:
            with sf.SoundFile(filepath) as f:
                print(f"{filename}: Sample rate = {f.samplerate} Hz")
        except RuntimeError as e:
            print(f"Không đọc được {filename}: {e}")