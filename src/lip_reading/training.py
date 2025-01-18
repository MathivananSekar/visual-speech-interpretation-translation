from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import json

# Ensure VSRDataset and VSRModel are correctly imported
from dataset import VSRDataset
from model import VSRModel

# Prepare DataLoader
video_dir = "../../data/datasets/s1/videos/"
audio_dir = "../../data/datasets/s1/audios/"
align_dir = "../../data/datasets/s1/alignments"
word_to_idx_path = "../../data/datasets/s1/word_to_idx.json"

video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mpg")]
audio_paths = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
align_files = [os.path.join(align_dir, f) for f in os.listdir(align_dir) if f.endswith(".align")]

dataset = VSRDataset(
    video_paths=video_paths,
    audio_paths=audio_paths,
    align_files=align_files,
    word_to_idx_path=word_to_idx_path,
    max_frames=75,
    sr=16000,
    fixed_audio_length=160
)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

with open(word_to_idx_path, "r") as f:
    word_to_idx = json.load(f)
num_classes = len(word_to_idx)

# Model, Loss, Optimizer
model = VSRModel(n_mfcc=160, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):  # Example: 10 epochs
    for video, audio, labels in data_loader:  # Assuming labels are part of the dataset
        optimizer.zero_grad()
        outputs = model(video, audio)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
