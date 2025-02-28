# data_loader_av_align.py

import os
import glob
import torch
from torch.utils.data import DataLoader, ConcatDataset
from src.data.dataset_av_align import AVAlignDataset
from functools import partial

def av_align_collate_fn(batch, pad_id=0):
    """
    batch: list of (vid_tensor, audio_tensor, align_tensor, text_tensor)
    We'll keep video/audio/align as a list for now, only pad text.
    """
    vids, auds, aligns, texts = zip(*batch)
    lengths = [len(t) for t in texts]
    max_len = max(lengths)
    B = len(texts)

    text_batch = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(texts):
        text_batch[i, :len(seq)] = seq

    return vids, auds, aligns, text_batch, torch.tensor(lengths, dtype=torch.long)

def collate_fn(batch, vocab):
    pad_id = vocab.pad_id if vocab.pad_id is not None else 0
    return av_align_collate_fn(batch, pad_id=pad_id)

def create_av_align_dataloader(
    processed_dir,
    vocab,
    batch_size=4,
    shuffle=True,
    num_workers=0
):
    npy_videos = glob.glob(os.path.join(processed_dir, "*_cropped.npy"))
    data_list = []
    for vid_path in npy_videos:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]  # e.g. vid1_cropped
        aud_path  = os.path.join(processed_dir, base_name.replace("_cropped","_audio") + ".npy")
        ali_path  = os.path.join(processed_dir, base_name.replace("_cropped","_align") + ".npy")
        txt_path  = os.path.join(processed_dir, base_name.replace("_cropped","_transcript") + ".txt")
        if os.path.exists(aud_path) and os.path.exists(ali_path) and os.path.exists(txt_path):
            data_list.append((vid_path, aud_path, ali_path, txt_path))

    dataset = AVAlignDataset(data_list=data_list, vocab=vocab)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: av_align_collate_fn(b, pad_id=vocab.pad_id if vocab.pad_id is not None else 0)
    )
    return loader

def gather_all_speakers_data(
    speaker_ids,
    base_path,
    vocab,
    batch_size=4,
    shuffle=True,
    num_workers=0
):
    datasets = []
    for spk in speaker_ids:
        proc_dir = os.path.join(base_path, "processed", spk)
        npy_videos = glob.glob(os.path.join(proc_dir, "*_cropped.npy"))
        data_list = []
        for vid_path in npy_videos:
            # print(f"Processing {vid_path}")
            base_name = os.path.splitext(os.path.basename(vid_path))[0]
            aud_path  = os.path.join(proc_dir, base_name.replace("_cropped","_audio") + ".npy")
            ali_path  = os.path.join(proc_dir, base_name.replace("_cropped","_align") + ".npy")
            txt_path  = os.path.join(proc_dir, base_name.replace("_cropped","_transcript") + ".txt")
            if os.path.exists(aud_path) and os.path.exists(ali_path) and os.path.exists(txt_path):
                data_list.append((vid_path, aud_path, ali_path, txt_path))
        ds = AVAlignDataset(data_list=data_list, vocab=vocab)
        datasets.append(ds)

    from torch.utils.data import ConcatDataset
    merged = ConcatDataset(datasets)
    loader = DataLoader(
        merged,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, vocab=vocab)
    )
    return loader
