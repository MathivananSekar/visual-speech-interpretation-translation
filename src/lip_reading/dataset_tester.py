from torch.utils.data import DataLoader
import os 
import torch.nn as nn
from dataset import VSRDataset


# Example file paths
video_dir = "../../data/datasets/s1/videos/"
audio_dir = "../../data/datasets/s1/audios/"
align_dir = "../../data/datasets/s1/alignments"

video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mpg")]
audio_paths = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
align_files = [os.path.join(align_dir, f) for f in os.listdir(align_dir) if f.endswith(".align")]

# Create dataset and dataloader
dataset = VSRDataset(video_paths, audio_paths, align_files)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3))

for batch in dataloader:
    video_tensors, audio_tensors, alignments = batch
    print("Input Video Tensor Shape:", video_tensors.shape)  # [batch_size, channels, num_frames, height, width]

    # Pass through the model
    output = model(video_tensors)
    print("Output Shape:", output.shape)
    break
