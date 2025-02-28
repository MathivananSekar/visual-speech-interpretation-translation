# dataset_av_align.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AVAlignDataset(Dataset):
    """
    Each sample has:
      - <vid>_cropped.npy => [T, H, W, C] for video
      - <vid>_audio.npy   => [n_mels, time_a] for audio
      - <vid>_align.npy   => [time_a] for alignment labels
      - <vid>_transcript.txt => text
    We return (video_tensor, audio_tensor, align_tensor, text_tensor).
    """
    def __init__(self, data_list, vocab, add_sos_eos=True):
        """
        data_list: list of (vid_path, aud_path, ali_path, txt_path)
        vocab: Vocab for transcripts
        add_sos_eos: Whether to wrap transcripts with <sos>/<eos>.
        """
        self.data_list = data_list
        self.vocab = vocab
        self.add_sos_eos = add_sos_eos

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vid_path, aud_path, ali_path, txt_path = self.data_list[idx]

        # Video
        frames_array = np.load(vid_path)  # [T, H, W, C]
        frames_tensor = torch.from_numpy(frames_array).float()
        if frames_tensor.max() > 1.0:
            frames_tensor /= 255.0
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # => [C, T, H, W]

        # Audio
        audio_array = np.load(aud_path)  # [n_mels, time_a]
        audio_tensor = torch.from_numpy(audio_array).float()

        # Align
        align_array = np.load(ali_path)  # [time_a]
        align_tensor = torch.from_numpy(align_array).long()

        # Transcript
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_line = f.read().strip()
        tokens = text_line.split()
        token_ids = [self.vocab.token_to_id(t.lower()) for t in tokens]
        if self.add_sos_eos:
            token_ids = [self.vocab.sos_id] + token_ids + [self.vocab.eos_id]
        text_tensor = torch.tensor(token_ids, dtype=torch.long)

        return frames_tensor, audio_tensor, align_tensor, text_tensor
