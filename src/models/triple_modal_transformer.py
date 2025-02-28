# triple_modal_transformer.py

import torch
import torch.nn as nn

class TripleModalBlock(nn.Module):
    """
    One layer that updates (V, A, L) => (video, audio, alignment).
    Each stream: self-attn, cross-attn with the other streams, feed-forward.
    """
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Video sub-layers
        self.video_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.video_cross_attn_audio = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.video_cross_attn_align = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.video_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.video_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

        # Audio sub-layers
        self.audio_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.audio_cross_attn_video = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.audio_cross_attn_align = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.audio_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.audio_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

        # Align sub-layers
        self.align_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.align_cross_attn_video = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.align_cross_attn_audio = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.align_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.align_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, V, A, L):
        # V => [B, T_v, d_model]
        # A => [B, T_a, d_model]
        # L => [B, T_l, d_model]

        # ----- VIDEO -----
        # 1) self-attn
        V2, _ = self.video_self_attn(V, V, V)
        V = self.video_norms[0](V + self.dropout(V2))
        # 2) cross-attn with A
        V3, _ = self.video_cross_attn_audio(V, A, A)
        V = V + self.dropout(V3)
        # cross-attn with L
        V4, _ = self.video_cross_attn_align(V, L, L)
        V = self.video_norms[1](V + self.dropout(V4))
        # 3) FFN
        V5 = self.video_ffn(V)
        V = self.video_norms[2](V + self.dropout(V5))

        # ----- AUDIO -----
        A2, _ = self.audio_self_attn(A, A, A)
        A = self.audio_norms[0](A + self.dropout(A2))
        A3, _ = self.audio_cross_attn_video(A, V, V)
        A = A + self.dropout(A3)
        A4, _ = self.audio_cross_attn_align(A, L, L)
        A = self.audio_norms[1](A + self.dropout(A4))
        A5 = self.audio_ffn(A)
        A = self.audio_norms[2](A + self.dropout(A5))

        # ----- ALIGN -----
        L2, _ = self.align_self_attn(L, L, L)
        L = self.align_norms[0](L + self.dropout(L2))
        L3, _ = self.align_cross_attn_video(L, V, V)
        L = L + self.dropout(L3)
        L4, _ = self.align_cross_attn_audio(L, A, A)
        L = self.align_norms[1](L + self.dropout(L4))
        L5 = self.align_ffn(L)
        L = self.align_norms[2](L + self.dropout(L5))

        return V, A, L

class TripleModalEncoder(nn.Module):
    """
    Stacks multiple TripleModalBlocks to refine (V, A, L).
    """
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TripleModalBlock(d_model, nhead, dim_feedforward, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, V, A, L):
        for layer in self.layers:
            V, A, L = layer(V, A, L)
        return V, A, L
