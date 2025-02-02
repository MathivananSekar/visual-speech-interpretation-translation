import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data.vocab import Vocab
from src.data.dataset import LipReadingDataset

def lipreading_collate_fn(batch):
    """
    Custom collate function to handle:
      - Stacking video tensors: (B, C, T, H, W)
      - Padding the token (label) sequences to the same length.
    Each item in batch is a tuple (frames, token_ids), where:
      - frames is a tensor of shape (T, 3, 64, 128)
      - token_ids is a list (or sequence) of token IDs (length T)
    """
    frames_list, token_ids_list = zip(*batch)
    batch_size = len(frames_list)

    # Find the maximum temporal length (number of frames) in the batch.
    # (Assuming each sample’s token_ids length equals its frame count.)
    seq_lengths = [f.shape[0] for f in frames_list]  # raw frame counts for each sample
    max_frames = max(seq_lengths)

    # Pad frames: pad along the time dimension.
    # Each frame tensor is of shape (T, 3, 64, 128); pad along T.
    padded_frames_list = [
        F.pad(frames, (0, 0, 0, 0, 0, max_frames - frames.shape[0])) 
        for frames in frames_list
    ]
    # Stack into tensor with shape (B, T, 3, 64, 128)
    frames_tensor = torch.stack(padded_frames_list, dim=0)

    # Pad token sequences to the maximum length.
    lengths = [len(seq) for seq in token_ids_list]  # these are the raw token counts
    max_len = max(lengths)
    text_batch = torch.full((batch_size, max_len), 0, dtype=torch.long)  # using 0 as pad_id

    for i, seq in enumerate(token_ids_list):
        text_batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    # Return the raw lengths as label_lengths and frame_lengths.
    # (If each frame has one label, these are the same.)
    frame_lengths = seq_lengths  # raw number of frames for each sample
    label_lengths = lengths      # raw number of tokens for each sample

    return frames_tensor, text_batch, frame_lengths, label_lengths



def create_dataloader(
    data_dir,
    vocab,
    batch_size=2,
    shuffle=True,
    add_sos_eos=True,
    num_workers=0
):
    dataset = LipReadingDataset(data_dir=data_dir, vocab=vocab)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lipreading_collate_fn
    )

    return dataloader


def gather_all_speakers_data(speaker_ids, base_path, vocab, batch_size, shuffle=True, num_workers=0):
    # Build a list of all .npz paths from each speaker subfolder
    all_npz_paths = []
    for spk_id in speaker_ids:
        dir_spk = os.path.join(base_path, "processed", spk_id)
        for f in os.listdir(dir_spk):
            if f.endswith(".npz"):
                all_npz_paths.append(os.path.join(dir_spk, f))

    # Create an "on-the-fly" dataset that can read these NPZs, or copy them into one folder
    # Simplest: store them in memory, or define a single dataset class that accepts a list of paths
    from torch.utils.data import Dataset

    class CombinedLipreadingDataset(Dataset):
        def __init__(self, npz_paths, vocab):
            super().__init__()
            self.npz_paths = npz_paths
            self.vocab = vocab

        def __len__(self):
            return len(self.npz_paths)

        def __getitem__(self, idx):
            npz_file = self.npz_paths[idx]
            data = np.load(npz_file, allow_pickle=True)
            frames = torch.tensor(data["frames"], dtype=torch.float32) / 255.0
            frames = frames.permute(0, 3, 1, 2)  # (T,3,64,128)
            labels_str = data["labels"]         # (T,) of strings
            labels_idx = [self.vocab.get_index(w) for w in labels_str]
            labels_idx = torch.tensor(labels_idx, dtype=torch.long)
            return frames, labels_idx

    dataset = CombinedLipreadingDataset(all_npz_paths, vocab)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lipreading_collate_fn
    )
    return loader


def load_vocab_from_json(json_path):
    """
    Load a dictionary from a JSON file {token: index, ...}, then
    build a Vocab object with sorted tokens by index order.
    """
    with open(json_path, "r") as f:
        word_dict = json.load(f)

    # Sort by index to ensure tokens align with their given indices
    sorted_items = sorted(word_dict.items(), key=lambda x: x[1])
    tokens = [word for word, idx in sorted_items]

    # If you have special tokens, define them here
    specials = {
        "pad": "<pad>",
        "unk": "<unk>",
        "sos": "<sos>",
        "eos": "<eos>"
    }

    vocab = Vocab(tokens=tokens, specials=specials)
    return vocab
