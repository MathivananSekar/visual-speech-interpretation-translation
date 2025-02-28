# alignment_encoder.py

import torch
import torch.nn as nn

class AlignmentEncoder(nn.Module):
    """
    If align_array is [time_a] for each sample, each element is an integer label
    representing the alignment word/phone.
    We'll embed them into a d_model dimension.
    """
    def __init__(self, align_vocab_size=100, d_model=256):
        super().__init__()
        self.embed = nn.Embedding(align_vocab_size, d_model)

    def forward(self, align_ids):
        # align_ids: [B, T_align]
        return self.embed(align_ids)  # => [B, T_align, d_model]
