import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbones import LipReadingModel
from src.models.positional_encoding import PositionalEncoding

class LipReading3DTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, max_len=250, dropout=0.1):
        super().__init__()
        
        # 1) Feature extractor backbone
        self.visual_backbone = LipReadingModel(
            num_classes=vocab_size,
            vocab_size=vocab_size,
            hidden_dim=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            return_sequence=True
        )
        self.feature_dim = d_model  # LipReadingModel outputs [B, T', hidden_dim]
        
        # 2) Project backbone features to d_model (identity if already matching)
        self.visual_fc = nn.Linear(self.feature_dim, d_model)  # Should be redundant if hidden_dim = d_model
        
        # 3) Positional encoding for visual embeddings (encoder)
        self.pos_encoder_vis = PositionalEncoding(d_model, dropout, max_len, batch_first=True)
        
        # 4) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 5) CTC Head
        self.ctc_head = nn.Linear(d_model, vocab_size)  # Added for CTC output
        
        # 6) Token embedding for decoder input (text)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.ln_emb = nn.LayerNorm(d_model)
        
        # 7) Positional encoding for token embeddings (decoder)
        self.pos_encoder_txt = PositionalEncoding(d_model, dropout, max_len, batch_first=True)
        
        # 8) Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 9) Final linear layer for attention output
        self.output_fc = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.vocab_size = vocab_size

    def encode_video(self, video_frames):
        """
        Args:
            video_frames: [B, T, 3, H, W]
        Returns:
            memory: [B, T', d_model]
        """
        feats = self.visual_backbone(video_frames)  # [B, T', hidden_dim]
        feats = self.visual_fc(feats)  # [B, T', d_model]
        feats = self.pos_encoder_vis(feats)  # [B, T', d_model]
        memory = self.encoder(feats)  # [B, T', d_model]
        return memory

    def decode_text(self, tgt_tokens, memory):
        """
        Args:
            tgt_tokens: [B, L]
            memory: [B, T', d_model]
        Returns:
            logits: [B, L, vocab_size]
        """
        emb = self.token_embedding(tgt_tokens)  # [B, L, d_model]
        emb = self.ln_emb(emb)
        emb = self.pos_encoder_txt(emb)  # [B, L, d_model]
        L = tgt_tokens.size(1)
        tgt_mask = self.generate_subsequent_mask(L).to(tgt_tokens.device)
        decoded = self.decoder(emb, memory, tgt_mask=tgt_mask)  # [B, L, d_model]
        logits = self.output_fc(decoded)  # [B, L, vocab_size]
        return logits

    def forward(self, video_frames, tgt_tokens):
        """
        Args:
            video_frames: [B, T, 3, H, W]
            tgt_tokens: [B, L]
        Returns:
            ctc_logits: [B, T', vocab_size]
            attn_logits: [B, L, vocab_size]
        """
        memory = self.encode_video(video_frames)  # [B, T', d_model]
        ctc_logits = self.ctc_head(memory)  # [B, T', vocab_size]
        attn_logits = self.decode_text(tgt_tokens, memory)  # [B, L, vocab_size]
        return ctc_logits, attn_logits

    def generate_subsequent_mask(self, size):
        """
        Shape: [size, size]
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask