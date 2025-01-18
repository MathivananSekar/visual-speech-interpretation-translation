import torch
from torch.utils.data import Dataset
import cv2
import librosa
import numpy as np
import json
import os

class VSRDataset(Dataset):
    def __init__(
        self,
        video_paths,
        audio_paths,
        align_files,
        word_to_idx_path,      # Path to the JSON file with your word_to_idx dictionary
        max_frames=75,
        sr=16000,
        fixed_audio_length=160
    ):
        """
        Args:
            video_paths (list): List of paths to video files.
            audio_paths (list): List of paths to corresponding audio files.
            align_files (list): List of alignment files.
            word_to_idx_path (str): Path to the JSON file containing word->index mappings.
            max_frames (int): Maximum number of video frames to keep.
            sr (int): Sampling rate for audio.
            fixed_audio_length (int): Number of MFCC time steps to keep (pad or truncate).
        """
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.align_files = align_files
        self.max_frames = max_frames
        self.sr = sr
        self.fixed_audio_length = fixed_audio_length

        # Load the word->index dictionary
        if not os.path.isfile(word_to_idx_path):
            raise FileNotFoundError(f"word_to_idx file not found: {word_to_idx_path}")

        with open(word_to_idx_path, "r") as f:
            self.word_to_idx = json.load(f)

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, video_path):
        """
        Load and preprocess video frames from the given video_path.
        - Resizes frames to 224x224.
        - Truncates/pads if frames exceed max_frames.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize each frame to (224, 224)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        frames = np.stack(frames, axis=0)
        if len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        elif len(frames) < self.max_frames:
            # Optional: zero-pad if fewer than max_frames
            pad_shape = (self.max_frames - len(frames), 224, 224, 3)
            padding = np.zeros(pad_shape, dtype=np.float32)
            frames = np.concatenate((frames, padding), axis=0)

        return frames

    def extract_mfcc(self, audio_path, sr=16000, n_mfcc=160, fixed_audio_length=160):
        """
        Instance method to extract MFCC features from an audio file.
        - Ensures we have n_mfcc features.
        - Pads or truncates the time dimension to fixed_audio_length.
        """
        # Load audio at the specified sampling rate
        audio, _ = librosa.load(audio_path, sr=sr)

        # Extract MFCC features: shape is [n_mfcc, time_frames]
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_mels=n_mfcc  # Matching n_mels to n_mfcc to allow 160 MFCC coefficients
        )

        # Check how many frames (time dimension)
        num_time_frames = mfcc.shape[1]

        # Pad or truncate the MFCC time dimension
        if num_time_frames < fixed_audio_length:
            # Create a zero matrix of shape [n_mfcc, fixed_audio_length]
            padded_mfcc = np.zeros((n_mfcc, fixed_audio_length), dtype=np.float32)
            padded_mfcc[:, :num_time_frames] = mfcc
            mfcc = padded_mfcc
        else:
            # Truncate to fixed_audio_length frames
            mfcc = mfcc[:, :fixed_audio_length]

        return mfcc

    def _load_audio(self, audio_path):
        """
        Load audio and convert to MFCC features using the instance method extract_mfcc.
        """
        audio_mfcc = self.extract_mfcc(
            audio_path,
            sr=self.sr,
            n_mfcc=160,
            fixed_audio_length=self.fixed_audio_length
        )
        return audio_mfcc

    def _parse_label(self, align_file):
        """
        Parse the alignment file and return a single integer label.
        Here, we pick the first token that isn't 'sil' as an example.
        Modify as needed for your labeling scheme.
        """
        with open(align_file, "r") as f:
            for line in f:
                start_str, end_str, token = line.strip().split()
                if token in self.word_to_idx and token != "sil":
                    return self.word_to_idx[token]

        # If all tokens are 'sil' or none found, fallback to 'sil' index (or 0 if not defined)
        return self.word_to_idx.get("sil", 0)

    def __getitem__(self, idx):
        """
        Return the (video_tensor, audio_tensor, label_tensor) tuple for a given index.
        """
        video_path = self.video_paths[idx]
        audio_path = self.audio_paths[idx]
        align_file = self.align_files[idx]

        # 1. Video
        video = self._load_video(video_path)
        video = video / 255.0  # Normalize pixel values
        video_tensor = torch.tensor(video, dtype=torch.float32).permute(3, 0, 1, 2)
        # shape: [3, max_frames, 224, 224]

        # 2. Audio
        audio_mfcc = self._load_audio(audio_path)
        audio_mfcc = (audio_mfcc - np.mean(audio_mfcc)) / (np.std(audio_mfcc) + 1e-6)
        audio_tensor = torch.tensor(audio_mfcc, dtype=torch.float32)
        # shape: [160, fixed_audio_length]

        # 3. Label
        label_id = self._parse_label(align_file)
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        return video_tensor, audio_tensor, label_tensor
