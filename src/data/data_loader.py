import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.data.vocab import Vocab
from src.data.dataset import LipReadingDataset

def lipreading_collate_fn(batch):
    frames_list, token_ids_list = zip(*batch)
    batch_size = len(frames_list)
    # print(f"Batch size: {batch_size}")
    
    for i, frames in enumerate(frames_list):
        # print(f"Sample {i} before padding: {frames.shape}")
        if frames.shape[1:] != (3, 64, 128):
            raise ValueError(f"Expected (T, 3, 64, 128), got {frames.shape} at index {i}")
    
    max_frames = max(frames.shape[0] for frames in frames_list)
    # print(f"Max frames in batch: {max_frames}")
    
    padded_frames_list = [
        F.pad(frames, (0, 0, 0, 0, 0, 0, 0, max_frames - frames.shape[0])) 
        for frames in frames_list
    ]
    
    for i, padded in enumerate(padded_frames_list):
        # print(f"Sample {i} after padding: {padded.shape}")
        if padded.shape != (max_frames, 3, 64, 128):
            raise ValueError(f"Expected ({max_frames}, 3, 64, 128), got {padded.shape} at index {i}")
    
    frames_tensor = torch.stack(padded_frames_list, dim=0)
    # print(f"Stacked frames tensor: {frames_tensor.shape}")
    
    max_len = max(len(seq) for seq in token_ids_list)
    text_batch = torch.full((batch_size, max_len), 0, dtype=torch.long)
    for i, seq in enumerate(token_ids_list):
        text_batch[i, :len(seq)] = torch.as_tensor(seq, dtype=torch.long)
    
    frame_lengths = [f.shape[0] for f in frames_list]
    label_lengths = [len(seq) for seq in token_ids_list]
    
    return frames_tensor, text_batch, frame_lengths, label_lengths



def create_dataloader(
    data_dir,
    vocab,
    batch_size=2,
    shuffle=True,
    add_sos_eos=True,
    num_workers=0
):
    dataset = LipReadingDataset(data_dir=data_dir,max_frames=75, vocab=vocab)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lipreading_collate_fn
    )

    return dataloader

class CombinedLipreadingDataset(Dataset):
        def __init__(self, npz_paths, vocab):
            super().__init__()
            self.valid_paths = []
            self.vocab = vocab
            # Pre-scan each file to confirm 3 channels
            for path in npz_paths:
                try:
                    data = np.load(path, allow_pickle=True)
                    frames = data["frames"]
                    # Check last dimension for 3 channels
                    if frames.shape[-1] == 3:
                        self.valid_paths.append(path)
                    else:
                        print(f"[WARN] Skipping {path}: frames.shape[-1] = {frames.shape[-1]} != 3")
                except Exception as e:
                    print(f"[WARN] Could not load {path}, skipping. Error: {e}")

        def __len__(self):
            return len(self.valid_paths)

        def __getitem__(self, idx):
            npz_file = self.valid_paths[idx]
            data = np.load(npz_file, allow_pickle=True)
            frames = torch.tensor(data["frames"], dtype=torch.float32) / 255.0
            
            # print(f"[{idx}] Raw shape from {npz_file}: {frames.shape}")
            if frames.dim() != 4 or frames.shape[-1] != 3 or frames.shape[1:3] != (64, 128):
                raise ValueError(f"Expected (T, 64, 128, 3), got {frames.shape} in {npz_file}")
            
            frames = frames.permute(0, 3, 1, 2)  # Should be (T, 3, 64, 128)
            # print(f"[{idx}] After permute: {frames.shape}")
            
            labels_str = data["labels"]
            labels_idx = [self.vocab.token_to_id(w) for w in labels_str]
            labels_idx = torch.tensor(labels_idx, dtype=torch.long)
            return frames, labels_idx
        
def gather_all_speakers_data(speaker_ids, base_path, vocab, batch_size, shuffle=True, num_workers=0):
    all_npz_paths = []
    for spk_id in speaker_ids:
        dir_spk = os.path.join(base_path, "processed", spk_id)
        # print(f"Scanning {dir_spk}")
        for f in os.listdir(dir_spk):
            if f.endswith(".npz"):
                path = os.path.join(dir_spk, f)
                all_npz_paths.append(path)
                # print(f"Added {path}")
    
    dataset = CombinedLipreadingDataset(all_npz_paths, vocab)
    print(f"Dataset size: {len(dataset)}")
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
