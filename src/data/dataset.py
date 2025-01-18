import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LipReadingDataset(Dataset):
    """
    A PyTorch Dataset that loads:
      - A .npy file for mouth-cropped frames: shape [T, H, W, C]
      - A corresponding .txt (or inline) transcript to be tokenized
    """
    def __init__(self, data_list, vocab, add_sos_eos=True, transform=None):
        """
        Args:
            data_list: A list of tuples (video_path, transcript_path),
                       or a CSV structure referencing them.
            vocab:     An instance of Vocab for token<->ID mapping.
            add_sos_eos: Whether to prepend <sos> and append <eos> to the token IDs.
            transform: Optional transformations on the video frames (e.g., augmentations).
        """
        self.data_list = data_list
        self.vocab = vocab
        self.add_sos_eos = add_sos_eos
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_path, transcript_path = self.data_list[idx]

        # Load the .npy file: shape [T, H, W, C]
        frames_array = np.load(video_path)  # dtype could be uint8 or float
        # Convert to float tensor, if needed
        frames_tensor = torch.from_numpy(frames_array).float()
        # Transpose to [C, T, H, W] for PyTorch 3D CNN usage
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

        # Apply any additional transform if desired
        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        # Read transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Tokenize (example: naive space-splitting; adapt for chars or subwords)
        tokens = text.split()  # e.g., ["place", "blue", "at", ...]
        token_ids = []
        for tok in tokens:
            tid = self.vocab.token_to_id(tok.lower())  # or just tok
            token_ids.append(tid)

        # Optionally add <sos> / <eos>
        if self.add_sos_eos:
            if self.vocab.sos_id is not None:
                token_ids = [self.vocab.sos_id] + token_ids
            if self.vocab.eos_id is not None:
                token_ids.append(self.vocab.eos_id)

        # Convert token IDs to a tensor
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        return frames_tensor, token_ids
