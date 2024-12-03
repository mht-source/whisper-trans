import whisper
import torch
import os
import subprocess
import math
import warnings

# Wyłącz specyficzne ostrzeżenie dotyczące FP16 na CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Funkcja do załadowania pliku audio z pliku wideo i obliczenia jego długości
def get_audio_duration(video_path):
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-f", "null", "-"],
            stderr=subprocess.PIPE,  # Capture stderr to parse the duration
            text=True
        )
        duration_line = [x for x in result.stderr.splitlines() if "Duration" in x]

        if len(duration_line) == 0:
            raise ValueError(f"Could not determine duration for file: {video_path}. Please check the file format and content.")

        duration_str = duration_line[0].split(",")[0].replace("Duration: ", "").strip()
        h, m, s = duration_str.split(":")
        total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
        return total_seconds
    
    except Exception as e:
        print(f"Error occurred while getting audio duration: {e}")
        return None

# Funkcja do transkrypcji pliku
def transcribe(model, video_path, output_srt_path, device):
    audio_length = get_audio_duration(video_path)
    if audio_length is None:
        print(f"Skipping transcription for {video_path} due to error in getting audio duration.")
        return

    # Transkrybuj oryginalny tekst
    print(f"Transcribing {video_path}...")

    # Przenieś model na GPU, jeśli CUDA jest dostępne
    model = model.to(device)
    
    original_result = model.transcribe(video_path, verbose=False)

    # Zapisz oryginalną transkrypcję do pliku SRT
    with open(output_srt_path, "w", encoding="utf-8") as f:  # Zapisuj z użyciem UTF-8
        for i, segment in enumerate(original_result['segments']):
            start_time = segment['start']
            end_time = segment['end']
            start = f"{math.floor(start_time // 3600):02}:{math.floor((start_time % 3600) // 60):02}:{math.floor(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end = f"{math.floor(end_time // 3600):02}:{math.floor((end_time % 3600) // 60):02}:{math.floor(end_time % 60):02},{int((end_time % 1) * 1000):03}"

            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text']}\n\n")

    print(f"Transcription complete for {video_path}! Saved to {output_srt_path}")

# Funkcja do przetwarzania wszystkich plików wideo w folderze
def process_videos_in_folder(folder_path):
    # Sprawdź dostępność CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    # Załaduj model Whisper
    model = whisper.load_model("large").to(device)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')):
            video_path = os.path.join(folder_path, filename)
            output_srt_path = os.path.splitext(video_path)[0] + ".srt"
            transcribe(model, video_path, output_srt_path, device)

# Ścieżka do folderu z plikami wideo
folder_path = 'C:/Users/mahit/prawko'

# Przetwarzaj wszystkie pliki wideo w folderze
process_videos_in_folder(folder_path)
