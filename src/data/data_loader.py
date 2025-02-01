import os
import glob
import json
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
    max_frames = max(frames.shape[1] for frames in frames_list)

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

    # Default pad_id to 0 unless we find otherwise in the dataset’s vocab
    pad_id = 0
    # Try to look for vocab.pad_id if available
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
    print(f"Scanning {processed_dir} for data files...")
    npy_files = glob.glob(os.path.join(processed_dir, "*_cropped.npy"))
    data_list = []
    for npy_file in npy_files:
        print(f"Found {npy_file}")
        base_name = os.path.splitext(os.path.basename(npy_file))[0]  # e.g. "vid1_cropped"
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


def gather_all_speakers_data(speaker_ids, base_path, vocab, batch_size, shuffle=True, num_workers=0):
    """
    For each speaker in speaker_ids, call create_dataloader(...) 
    to gather its data_list. Then combine them all into one dataset+loader.
    """
    from src.data.dataset import LipReadingDataset
    from torch.utils.data import DataLoader
    import glob
    import os

    combined_data_list = []

    # For each speaker, gather the .npy/.txt pairs
    for spk in speaker_ids:
        processed_dir_spk = os.path.join(base_path, "processed", spk)
        npy_files = glob.glob(os.path.join(processed_dir_spk, "*_cropped.npy"))
        for npy_file in npy_files:
            base_name = os.path.splitext(os.path.basename(npy_file))[0]
            txt_name = base_name.replace("_cropped", "_transcript") + ".txt"
            txt_file = os.path.join(processed_dir_spk, txt_name)
            if os.path.exists(txt_file):
                combined_data_list.append((npy_file, txt_file))

    print(f"Total samples across all speakers: {len(combined_data_list)}")

    # Create a single dataset
    dataset = LipReadingDataset(
        data_list=combined_data_list,
        vocab=vocab,
        add_sos_eos=True,
        transform=None
    )

    # Create a single DataLoader from the combined dataset
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
    Load a dictionary from a JSON file {word: index, ...}, then
    build a Vocab object with sorted tokens in index order.
    """
    with open(json_path, "r") as f:
        word_dict = json.load(f)

    # Sort by index to ensure tokens align with their given indices
    sorted_items = sorted(word_dict.items(), key=lambda x: x[1])
    tokens = [word for word, idx in sorted_items]

    # Define any special tokens (if you want them appended or separate).
    # You may have to adjust their indices if you require them to match certain IDs.
    specials = {
        "pad": "<pad>",
        "unk": "<unk>",
        "sos": "<sos>",
        "eos": "<eos>"
    }

    vocab = Vocab(tokens=tokens, specials=specials)
    return vocab