import librosa
import numpy as np
import os 

processed_audio_dir = "../data/datasets/s1/processed_audios/"
mfcc_dir = "../data/datasets/s1/mfcc/"

os.makedirs(mfcc_dir, exist_ok=True)

def extract_mfcc(audio_path, save_path, n_mfcc=13, sr=16000):
    """
    Extract MFCC features from an audio file and save them.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Save the MFCC features as a .npy file
    np.save(save_path, mfcc)

# Process all audio files
for audio_file in os.listdir(processed_audio_dir):
    audio_path = os.path.join(processed_audio_dir, audio_file)
    mfcc_save_path = os.path.join(mfcc_dir, os.path.splitext(audio_file)[0] + ".npy")
    
    print(f"Processing {audio_file} for MFCC features...")
    extract_mfcc(audio_path, mfcc_save_path)
    print(f"MFCC features for {audio_file} saved at {mfcc_save_path}!")

