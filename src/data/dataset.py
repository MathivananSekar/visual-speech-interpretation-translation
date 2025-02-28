import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F_t

import random
import torch
import torchvision.transforms.functional as F_t

class ClipAugmentTransform:
    """
    Apply the same random spatial + color transforms across all frames in a clip
    to preserve mouth alignment.
    """

    def __init__(self, resize=(112, 112),
                 color_jitter=0.2,
                 horizontal_flip_prob=0.5):
        self.resize = resize
        self.color_jitter = color_jitter
        self.hflip_prob = horizontal_flip_prob

    def __call__(self, clip):
        """
        clip: [C, T, H, W], with pixel values in [0,1].
        """
        C, T, H, W = clip.shape

        do_flip = (random.random() < self.hflip_prob)
        # pick random brightness/contrast
        brightness_factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        contrast_factor   = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)

        # Possibly random crop if desired
        # For example, random shift up to +/- 5 px
        max_shift = 5
        shift_h = random.randint(-max_shift, max_shift)
        shift_w = random.randint(-max_shift, max_shift)

        transformed_frames = []
        for t in range(T):
            frame = clip[:, t, :, :]  # shape [C, H, W]
            # Horizontal flip
            if do_flip:
                frame = F_t.hflip(frame)
            # Shift (pad + crop) if you want
            frame = F_t.pad(frame, padding=max_shift, fill=0)
            frame = frame[:, (max_shift+shift_h):(max_shift+shift_h+H),
                             (max_shift+shift_w):(max_shift+shift_w+W)]
            # Resize
            frame = F_t.resize(frame, self.resize)
            # Color jitter
            frame = F_t.adjust_brightness(frame, brightness_factor)
            frame = F_t.adjust_contrast(frame, contrast_factor)
            transformed_frames.append(frame.unsqueeze(1))

        # Re-stack
        return torch.cat(transformed_frames, dim=1)



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
        self.transform = ClipAugmentTransform(resize=(112,112), color_jitter=0.2, horizontal_flip_prob=0.5)


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
