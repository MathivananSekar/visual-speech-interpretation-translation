import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbones import SpatioTemporalMouthNet  # Use our updated LipNet-inspired backbone
from src.models.positional_encoding import PositionalEncoding

class LipReading3DTransformer(nn.Module):
    """
    End-to-end lipreading architecture that integrates:
      1) A 3D CNN backbone (SpatioTemporalMouthNet) for spatiotemporal feature extraction.
      2) A Transformer Encoder to process visual embeddings.
      3) A Transformer Decoder for autoregressive text generation.
      4) A final linear layer mapping to vocabulary logits.
      
    This revised model uses a LipNet-inspired backbone, layer normalization,
    and dropout strategies to boost performance.
    """
    def __init__(self,
                 vocab_size,          # output vocabulary size
                 d_model=256,
                 nhead=4,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=1024,
                 max_len=250,
                 dropout=0.1):
        super().__init__()
        
        # 1) 3D CNN feature extractor (SpatioTemporalMouthNet)
        self.visual_backbone = SpatioTemporalMouthNet(return_sequence=True, dropout_prob=dropout)
        self.feature_dim = 96  # As defined in SpatioTemporalMouthNet, output channels from Block 3
        
        # 2) Project backbone features to d_model dimension
        self.visual_fc = nn.Linear(self.feature_dim, d_model)
        
        # 3) Positional encoding for visual embeddings (encoder)
        self.pos_encoder_vis = PositionalEncoding(d_model, dropout, max_len)
        
        # 4) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation="relu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 5) Token embedding for decoder input (text)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Optionally add layer normalization after token embedding:
        self.ln_emb = nn.LayerNorm(d_model)
        
        # 6) Positional encoding for token embeddings (decoder)
        self.pos_encoder_txt = PositionalEncoding(d_model, dropout, max_len, batch_first=True)
        
        # 7) Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation="relu")
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 8) Final linear layer: map decoder outputs to vocabulary logits
        self.output_fc = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.vocab_size = vocab_size

    def encode_video(self, video_frames):
        """
        Encode video frames using the 3D CNN backbone.
        
        Args:
            video_frames: Tensor of shape [B, C, T, H, W] (channel-first).
        Returns:
            memory: Tensor of shape [T', B, d_model] (visual embeddings with positional encoding).
        """
        # Extract spatiotemporal features: output shape [B, T', feature_dim]
        feats = self.visual_backbone(video_frames)
        # Project to d_model: [B, T', d_model]
        feats = self.visual_fc(feats)
        # Transpose to [T', B, d_model] for Transformer encoder
        feats = feats.transpose(0, 1)
        # Add positional encoding
        feats = self.pos_encoder_vis(feats)
        # Pass through Transformer encoder
        memory = self.encoder(feats)
        return memory

    def decode_text(self, tgt_tokens, memory):
        """
        Decode target tokens using Transformer decoder.
        
        Args:
            tgt_tokens: Tensor of shape [B, L] (token IDs for teacher forcing).
            memory: Encoder output, shape [T', B, d_model].
        Returns:
            logits: Tensor of shape [B, L, vocab_size].
        """
        # 1) Embed target tokens and apply layer normalization.
        emb = self.token_embedding(tgt_tokens)  # [B, L, d_model]
        emb = self.ln_emb(emb)
        # 2) Add positional encoding; note we use batch_first=True here.
        emb = self.pos_encoder_txt(emb)  # [B, L, d_model]
        # 3) Transpose to [L, B, d_model] as expected by Transformer decoder.
        emb = emb.transpose(0, 1)
        # 4) Create a causal mask for the decoder (prevent attention to future tokens)
        L = tgt_tokens.size(1)
        tgt_mask = self.generate_subsequent_mask(L).to(tgt_tokens.device)
        # 5) Run Transformer decoder. (No explicit key_padding_mask here; add if needed.)
        decoded = self.decoder(emb, memory, tgt_mask=tgt_mask)
        # 6) Transpose back to [B, L, d_model]
        decoded = decoded.transpose(0, 1)
        # 7) Project to vocabulary logits.
        logits = self.output_fc(decoded)
        return logits

    def forward(self, video_frames, tgt_tokens):
        """
        Forward pass of the model.
        
        Args:
            video_frames: [B, C, T, H, W] (input video clip).
            tgt_tokens: [B, L] (input tokens for teacher forcing).
        Returns:
            logits: [B, L, vocab_size] (vocabulary scores).
        """
        memory = self.encode_video(video_frames)
        logits = self.decode_text(tgt_tokens, memory)
        return logits

    def generate_subsequent_mask(self, size):
        """
        Generate a subsequent (causal) mask for the Transformer decoder.
        Shape: [size, size]
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
