import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LipReadingDataset(Dataset):
    """
    Expects each .npz file in data_dir to have:
      - frames: shape (num_frames, 64, 128, 3)
      - labels: shape (num_frames,)  [strings of words]
    """
    def __init__(self, data_dir, vocab, max_frames, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.vocab = vocab
        self.transform = transform
        self.max_frames = max_frames
        # Collect all .npz files in data_dir
        self.file_list = [
            f for f in os.listdir(data_dir) if f.endswith(".npz")
        ]
        self.file_list.sort()  # optional, for consistent ordering
        # Calculate max_frames dynamically
        self.max_frames = max(
            np.load(os.path.join(self.data_dir, f))["frames"].shape[0]
            for f in self.file_list
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_file = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(npz_file, allow_pickle=True)
        frames = data["frames"]           # shape (num_frames, 64, 128, 3)
        # Ensure frames have 3 channels
        if frames.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, but got {frames.shape[-1]} channels in file {npz_file}")
        labels = data["labels"].tolist()       # labels is a list of strings
        # print("Inside LipReadingDataset - Raw labels (string):", labels)
        # Convert frames to float32, normalize to [0,1], then to Tensor
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        # Reorder to (num_frames, 3, 64, 128)
        frames = frames.permute(0, 3, 1, 2)
        if frames.shape[0] < self.max_frames:
            padding = torch.zeros((self.max_frames - frames.shape[0], *frames.shape[1:]))
            frames = torch.cat([frames, padding], dim=0)
        else:
            frames = frames[:self.max_frames]

        # Map words -> integer indices
        label_indices = [self.vocab.token_to_id(str(word)) for word in labels]
        label_indices = torch.tensor(label_indices, dtype=torch.long)  # (num_frames,)

        # Optionally apply transforms
        if self.transform is not None:
            frames = self.transform(frames)

        return frames, label_indices
