# model_av_align_cross.py

import torch
import torch.nn as nn
from src.models.backbones_3d import VideoEncoder3D
from src.models.audio_encoder import AudioEncoder1D
from src.models.alignment_encoder import AlignmentEncoder
from src.models.triple_modal_transformer import TripleModalEncoder
from src.models.positional_encoding import PositionalEncoding

class AVAlignCrossModalModel(nn.Module):
    """
    1) video_enc -> [B, T_v, d_model]
    2) audio_enc -> [B, T_a, d_model]
    3) align_enc -> [B, T_l, d_model]
    4) triple-stream cross attn (num_layers)
    5) fuse => e.g. concat => [B, T_v+T_a+T_l, d_model]
    6) transformer decoder => text
    """
    def __init__(self,
                 vocab_size,
                 d_model=256,
                 nhead=4,
                 num_encoder_layers=2,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_len=1000,
                 num_decoder_layers=4,
                 align_vocab_size=100):
        super().__init__()

        # Encoders
        self.video_enc = VideoEncoder3D(output_dim=d_model)
        self.audio_enc = AudioEncoder1D(output_dim=d_model)
        self.align_enc = AlignmentEncoder(align_vocab_size=align_vocab_size, d_model=d_model)

        # triple cross
        self.triple_encoder = TripleModalEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_encoder_layers
        )

        # final fuse + pos embed
        self.fuse_pos_enc = PositionalEncoding(d_model, dropout, max_len, batch_first=True)

        # Decoder
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.dec_pos_enc = PositionalEncoding(d_model, dropout, max_len, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers=num_decoder_layers
        )
        self.out_fc = nn.Linear(d_model, vocab_size)

    def forward(self, video, audio, align_ids, tgt_tokens, pad_id=0):
        """
        video: [B, 3, T_v, H, W]
        audio: [B, n_mels, T_a]
        align_ids: [B, T_l]
        tgt_tokens: [B, L]
        """
        # 1) encode each
        V = self.video_enc(video)         # => [B, T_v, d_model]
        A = self.audio_enc(audio)         # => [B, T_a, d_model]
        L = self.align_enc(align_ids)     # => [B, T_l, d_model]

        # 2) triple cross
        V, A, L = self.triple_encoder(V, A, L)  # => each is [B, T_v, d_model], etc.

        # 3) fuse => concat
        fused = torch.cat([V, A, L], dim=1)  # => [B, T_v+T_a+T_l, d_model]
        fused = self.fuse_pos_enc(fused)

        # 4) decode text
        emb = self.token_embed(tgt_tokens)
        emb = self.dec_pos_enc(emb)

        L_dec = tgt_tokens.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L_dec).to(tgt_tokens.device)
        tgt_mask = (tgt_mask == float('-inf'))  # boolean
        tgt_key_padding = (tgt_tokens == pad_id)

        out = self.transformer_decoder(
            emb, fused,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding
        )
        logits = self.out_fc(out)  # => [B, L, vocab_size]
        return logits
