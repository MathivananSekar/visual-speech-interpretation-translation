from pydub import AudioSegment
import os

audio_dir = "../data/datasets/s1/audios/"
processed_audio_dir = "../data/datasets/s1/processed_audios/"
os.makedirs(processed_audio_dir, exist_ok=True)

def process_audio(audio_file, output_path):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")

for audio_file in os.listdir(audio_dir):
    if not audio_file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
        continue
    input_path = os.path.join(audio_dir, audio_file)
    output_path = os.path.join(processed_audio_dir, audio_file)
    process_audio(input_path, output_path)
