import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbone_r2plus1d import SpatioTemporalR2Plus1D
from src.models.positional_encoding import PositionalEncoding

class LipReadingR2Plus1DTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model=384,           # bigger hidden dimension
                 nhead=6,
                 num_encoder_layers=6,  # deeper
                 num_decoder_layers=6,
                 dim_feedforward=1536,
                 max_len=400,
                 dropout=0.3,
                 use_ctc=False):
        super().__init__()
        self.visual_backbone = SpatioTemporalR2Plus1D(return_sequence=True)
        self.feature_dim = 512  # R(2+1)D final channel
        self.visual_fc = nn.Linear(self.feature_dim, d_model)

        self.pos_encoder_vis = PositionalEncoding(d_model, dropout, max_len, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_txt = PositionalEncoding(d_model, dropout, max_len, batch_first=True)
        self.output_fc = nn.Linear(d_model, vocab_size)

        self.use_ctc = use_ctc
        if use_ctc:
            # +1 for blank
            self.ctc_vocab_size = vocab_size + 1
            self.ctc_fc = nn.Linear(d_model, self.ctc_vocab_size)

        self._reset_parameters()

        self.d_model = d_model
        self.vocab_size = vocab_size

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def encode_video(self, video_frames):
        """
        video_frames: [B, 3, T, H, W]
        Returns memory: [B, T_enc, d_model]
        """
        feats = self.visual_backbone(video_frames)  # [B, T_enc, 512]
        feats = self.visual_fc(feats)               # [B, T_enc, d_model]
        feats = self.pos_encoder_vis(feats)         # add positional enc
        memory = self.transformer.encoder(feats)    # [B, T_enc, d_model]
        return memory

    def decode_text(self, tgt_tokens, memory, pad_id=0):
        """
        tgt_tokens: [B, L]
        memory: [B, T_enc, d_model]
        Returns: [B, L, vocab_size]
        """
        emb = self.token_embedding(tgt_tokens)        # [B, L, d_model]
        emb = self.pos_encoder_txt(emb)

        L = tgt_tokens.shape[1]
        float_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tgt_tokens.device)
        # Convert float_mask => bool where True => blocked
        tgt_mask = (float_mask == float('-inf'))
        # Key padding mask: True => ignore
        tgt_key_padding_mask = (tgt_tokens == pad_id)

        output = self.transformer.decoder(
            emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        logits = self.output_fc(output)  # [B, L, vocab_size]
        return logits

    def forward(self, video_frames, tgt_tokens=None, pad_id=0):
        """
        If training seq2seq, pass in tgt_tokens with teacher forcing.
        """
        memory = self.encode_video(video_frames)
        outputs = {}

        if tgt_tokens is not None:
            seq2seq_logits = self.decode_text(tgt_tokens, memory, pad_id=pad_id)
            outputs['seq2seq_logits'] = seq2seq_logits

        if self.use_ctc:
            ctc_logits = self.ctc_fc(memory)   # [B, T_enc, ctc_vocab_size]
            outputs['ctc_logits'] = ctc_logits

        return outputs
