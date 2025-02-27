import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F_t

class ClipTransform:
    """
    Applies a single random crop to all frames in a clip, then
    resizes, and applies mild color jitter to each frame consistently.
    """

    def __init__(self, crop_size=(100, 100), resize=(112, 112),
                 brightness=0.05, contrast=0.05):
        self.crop_size = crop_size
        self.resize = resize
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, clip):
        """
        clip: Tensor of shape (C, T, H, W), with C=3 (RGB).
        We will:
            - pick one random crop region (top, left)
            - apply it to each of the T frames
            - resize to final size
            - apply mild color jitter
        """
        C, T, H, W = clip.shape
        crop_h, crop_w = self.crop_size

        if H < crop_h or W < crop_w:
            # Fallback: just center-crop if the random crop won't fit
            top = max(0, (H - crop_h) // 2)
            left = max(0, (W - crop_w) // 2)
        else:
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)

        cropped_frames = []
        for t in range(T):
            frame = clip[:, t, :, :]  # shape [C, H, W]
            # Crop
            frame = frame[:, top:top+crop_h, left:left+crop_w]
            # Resize
            frame = F_t.resize(frame, self.resize)
            # Apply color jitter
            frame = F_t.adjust_brightness(frame, 1.0 + random.uniform(-self.brightness, self.brightness))
            frame = F_t.adjust_contrast(frame, 1.0 + random.uniform(-self.contrast, self.contrast))

            cropped_frames.append(frame.unsqueeze(1))  # shape [C, 1, newH, newW]

        # Re-stack along time dimension
        transformed_clip = torch.cat(cropped_frames, dim=1)  # [C, T, newH, newW]
        return transformed_clip


class LipReadingDataset(Dataset):
    """
    A PyTorch Dataset that loads:
      - A .npy file for mouth-cropped frames: shape [T, H, W, C]
      - A corresponding .txt transcript to be tokenized
    """
    def __init__(self, data_list, vocab, add_sos_eos=True, transform=None):
        """
        Args:
            data_list: list of (video_path, transcript_path)
            vocab:     Vocab instance for token<->ID mapping
            add_sos_eos: Whether to prepend <sos> and append <eos> tokens
            transform: Optional transform that processes the entire clip
        """
        self.data_list = data_list
        self.vocab = vocab
        self.add_sos_eos = add_sos_eos
        # By default, use the new ClipTransform
        self.transform = transform if transform is not None else ClipTransform()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_path, transcript_path = self.data_list[idx]

        # Load frames: shape [T, H, W, C]
        frames_array = np.load(video_path)  # might be uint8
        frames_tensor = torch.from_numpy(frames_array).float()
        if frames_tensor.max() > 1.0:
            frames_tensor /= 255.0

        # Permute to [C, T, H, W]
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)

        # Apply transform (consistent random crop, resize, mild color jitter)
        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        # Read transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Tokenize by words (GRID corpus is small enough for that)
        tokens = text.split()
        token_ids = []
        for tok in tokens:
            tid = self.vocab.token_to_id(tok.lower())
            token_ids.append(tid)

        if self.add_sos_eos:
            if self.vocab.sos_id is not None:
                token_ids = [self.vocab.sos_id] + token_ids
            if self.vocab.eos_id is not None:
                token_ids.append(self.vocab.eos_id)

        token_ids = torch.tensor(token_ids, dtype=torch.long)

        return frames_tensor, token_ids
