import os
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.data.vocab import Vocab
from src.data.dataset import LipReadingDataset

def lipreading_collate_fn(batch):
    """
    Custom collate function to handle:
      - Stacking video tensors: (B, C, T, H, W)
      - Padding the text sequences to the same length
    batch: list of (frames_tensor, token_ids)
    """

    frames_list, token_ids_list = zip(*batch)
    
    # Find the maximum temporal size
    max_frames = max([frames.shape[1] for frames in frames_list])

    # Pad all tensors to have the same temporal size
    padded_frames_list = [
        F.pad(frames, (0, 0, 0, 0, 0, max_frames - frames.shape[1]))  # Pad along T
        for frames in frames_list
    ]

    # 1. Stack frames (video data) into a single tensor
    # Stack the tensors
    frames_tensor = torch.stack(padded_frames_list, dim=0)  # shape (B, C, T, H, W)

    # 2. Pad token sequences to same length
    lengths = [len(seq) for seq in token_ids_list]
    max_len = max(lengths)
    batch_size = len(token_ids_list)

    # We fill with pad_id
    # We'll assume the Vocab pad_id is 0 if not set. Adapt as needed.
    pad_id = 0
    if hasattr(batch[0][1], 'vocab') and batch[0][1].vocab.pad_id is not None:
        pad_id = batch[0][1].vocab.pad_id

    text_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

    for i, seq in enumerate(token_ids_list):
        text_batch[i, :len(seq)] = seq

    return frames_tensor, text_batch, lengths

def create_dataloader(
    processed_dir,
    vocab,
    batch_size=2,
    shuffle=True,
    add_sos_eos=True,
    num_workers=0
):
    """
    Scans a directory of processed files (npy + txt),
    builds a data_list, constructs a LipReadingDataset,
    and wraps it in a DataLoader with a custom collate_fn.
    """
    # Build data_list
    # We look for all *_cropped.npy in processed_dir,
    # and expect a matching *_transcript.txt
    print(f"Scanning {processed_dir} for data files...")
    npy_files = glob.glob(os.path.join(processed_dir, "*_cropped.npy"))
    data_list = []
    for npy_file in npy_files:
        print(f"Found {npy_file}")
        base_name = os.path.splitext(os.path.basename(npy_file))[0]  # e.g. "vid1_cropped"
        # transcript file might be "vid1_transcript.txt" if that’s your naming
        txt_name = base_name.replace("_cropped", "_transcript") + ".txt"
        txt_file = os.path.join(processed_dir, txt_name)
        if os.path.exists(txt_file):
            data_list.append((npy_file, txt_file))

    print(f"Found {len(data_list)} samples in {processed_dir}")

    dataset = LipReadingDataset(
        data_list=data_list,
        vocab=vocab,
        add_sos_eos=add_sos_eos,
        transform=None  # or your custom transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lipreading_collate_fn
    )

    return dataloader

if __name__ == "__main__":
    # Example usage
    # Suppose we built a vocab with some dummy tokens
    tokens = ["place", "blue", "at", "red", "green", "two", "one"]
    specials = {
        "pad": "<pad>",
        "unk": "<unk>",
        "sos": "<sos>",
        "eos": "<eos>"
    }
    vocab = Vocab(tokens=tokens, specials=specials)
    
    base_path = "data"
    speaker_id = "s1"
    # Path to the directory with processed .npy + .txt
    processed_dir = os.path.join(base_path, "processed", speaker_id)
    
    loader = create_dataloader(
        processed_dir=processed_dir,
        vocab=vocab,
        batch_size=2,
        shuffle=False
    )
    
    for batch_idx, (videos, texts, lengths) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(" - videos shape:", videos.shape)  # (B, C, T, H, W)
        print(" - texts shape:", texts.shape)    # (B, max_len)
        print(" - lengths:", lengths)            # list of actual seq lengths
        # Here you would feed (videos, texts) into your model
        break
