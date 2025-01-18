import librosa
import numpy as np

def extract_mfcc(audio_path, sr=16000, n_mfcc=160, fixed_audio_length=160):
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Extract MFCC features (output will be [n_mfcc, time_frames])
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_mels=160)
    print(f"MFCC shape: {mfcc.shape}")
    # Check number of time frames (columns in MFCC matrix)
    num_time_frames = mfcc.shape[1]
    print(f"Number of time frames: {num_time_frames}")

    # If there are fewer time frames than fixed_audio_length, pad the MFCC matrix
    if num_time_frames < fixed_audio_length:
        # Padding: pad columns (time frames) with zeros
        padded_mfcc = np.zeros((n_mfcc, fixed_audio_length), dtype=np.float32)
        padded_mfcc[:, :num_time_frames] = mfcc  # Fill with original MFCC
        num_time_frames_after_padding = padded_mfcc.shape[1]
        print(f"Number of time frames: {num_time_frames_after_padding}")
    else:
        # Truncate if there are too many frames
        padded_mfcc = mfcc[:, :fixed_audio_length]

    return padded_mfcc

# Example usage:
audio_path = '../../data/datasets/s1/audios/bbaf2n.wav'
mfcc_features = extract_mfcc(audio_path)
print(mfcc_features.shape)  # Should print (160, fixed_audio_length)
