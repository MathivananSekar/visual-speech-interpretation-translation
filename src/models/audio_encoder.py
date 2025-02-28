# audio_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder1D(nn.Module):
    """
    A 1D CNN that outputs a time-sequence from log-mel.
    Return shape [B, T_a, output_dim].
    """
    def __init__(self, n_mels=40, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.proj = nn.Linear(128, output_dim)

    def forward(self, x):
        # x => [B, n_mels, time_a]
        x = F.relu(self.conv1(x))  # => [B,128,time_a]
        x = F.relu(self.conv2(x))  # => [B,128,time_a]
        x = x.transpose(1, 2)      # => [B,time_a,128]
        x = self.proj(x)          # => [B,time_a,output_dim]
        return x
