# data_loader.py

import os
import glob
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.vocab import Vocab
from src.data.dataset import LipReadingDataset

class LipreadingCollator:
    """Collator class that can be pickled on Windows, avoiding local lambdas."""
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        """
        batch: list of (frames_tensor, token_ids)
        Returns: (frames_tensor, text_batch, lengths)
        """
        frames_list, token_ids_list = zip(*batch)

        # 1) Find max temporal length among videos
        max_frames = max(frames.shape[1] for frames in frames_list)

        # 2) Pad temporal dimension for video frames
        padded_frames_list = []
        for frames in frames_list:
            # frames shape = (C, T, H, W)
            diff = max_frames - frames.shape[1]
            # Pad along the time dimension
            # (left=0, right=0, top=0, bottom=0, front=0, back=diff)
            padded = F.pad(frames, (0, 0, 0, 0, 0, diff))  
            padded_frames_list.append(padded)

        frames_tensor = torch.stack(padded_frames_list, dim=0)  # (B, C, T, H, W)

        # 3) Pad token sequences
        lengths = [len(seq) for seq in token_ids_list]
        max_len = max(lengths)
        batch_size = len(token_ids_list)

        text_batch = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(token_ids_list):
            text_batch[i, :len(seq)] = seq

        return frames_tensor, text_batch, torch.tensor(lengths, dtype=torch.long)

def create_dataloader(
    processed_dir,
    vocab,
    batch_size=4,
    shuffle=True,
    add_sos_eos=True,
    num_workers=0
):
    """
    Scans a directory of processed files (npy + txt),
    constructs a LipReadingDataset, returns a DataLoader
    using the top-level LipreadingCollator.
    """
    npy_files = glob.glob(os.path.join(processed_dir, "*_cropped.npy"))
    data_list = []
    for npy_file in npy_files:
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        txt_name = base_name.replace("_cropped", "_transcript") + ".txt"
        txt_file = os.path.join(processed_dir, txt_name)
        if os.path.exists(txt_file):
            data_list.append((npy_file, txt_file))

    dataset = LipReadingDataset(
        data_list=data_list,
        vocab=vocab,
        add_sos_eos=add_sos_eos,
        transform=None
    )

    # Use the collator class instead of a lambda
    collator = LipreadingCollator(pad_id=vocab.pad_id if vocab.pad_id is not None else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator
    )
    return dataloader

def gather_all_speakers_data(speaker_ids, base_path, vocab, batch_size,
                             shuffle=True, num_workers=0):
    combined_data_list = []
    for spk in speaker_ids:
        processed_dir_spk = os.path.join(base_path, "processed", spk)
        npy_files = glob.glob(os.path.join(processed_dir_spk, "*_cropped.npy"))
        for npy_file in npy_files:
            base_name = os.path.splitext(os.path.basename(npy_file))[0]
            txt_name = base_name.replace("_cropped", "_transcript") + ".txt"
            txt_file = os.path.join(processed_dir_spk, txt_name)
            if os.path.exists(txt_file):
                combined_data_list.append((npy_file, txt_file))

    dataset = LipReadingDataset(
        data_list=combined_data_list,
        vocab=vocab,
        add_sos_eos=True,
        transform=None
    )

    collator = LipreadingCollator(pad_id=vocab.pad_id if vocab.pad_id is not None else 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator
    )
    return loader

def load_vocab_from_json(json_path):
    with open(json_path, "r") as f:
        word_dict = json.load(f)
    sorted_items = sorted(word_dict.items(), key=lambda x: x[1])
    tokens = [word for word, idx in sorted_items]

    specials = {
        'pad': '<pad>',
        'unk': '<unk>',
        'sos': '<sos>',
        'eos': '<eos>',
        'blank': '<blank>'
    }
    vocab = Vocab(tokens=tokens, specials=specials)
    return vocab
