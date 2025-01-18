import torch
import torch.nn as nn
from src.models.backbones import SpatioTemporalResNet
from src.models.positional_encoding import PositionalEncoding


class LipReading3DTransformer(nn.Module):
    """
    End-to-end architecture:
      1) 3D CNN backbone -> spatiotemporal features [B, T', 512]
      2) Transformer Encoder -> transforms visual embeddings
      3) Transformer Decoder -> autoregressively outputs text tokens
      4) Linear layer -> vocab logits
    """
    def __init__(self,
                 vocab_size,          # size of output tokens (characters/subwords)
                 d_model=256,
                 nhead=4,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=1024,
                 max_len=250,
                 dropout=0.1):
        super().__init__()
        
        # 3D CNN feature extractor
        self.visual_backbone = SpatioTemporalResNet(return_sequence=True)
        self.feature_dim = 512  # r3d_18 final channels
        
        # Project CNN features -> d_model
        self.visual_fc = nn.Linear(self.feature_dim, d_model)
        
        # Positional encoding for the visual embeddings (encoder)
        self.pos_encoder_vis = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer itself
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation="relu")
        
        # Token embedding for the decoder (text input)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding for token embeddings (decoder)
        self.pos_encoder_txt = PositionalEncoding(d_model, dropout, max_len)
        
        # Final linear -> vocab logits
        self.output_fc = nn.Linear(d_model, vocab_size)
        
        # We also store some parameters for usage
        self.d_model = d_model
        self.vocab_size = vocab_size

    def encode_video(self, video_frames):
        """
        video_frames: [B, C, T, H, W] (channel-first).
        Returns memory: [T', B, d_model]
        """
        # 1) Extract spatiotemporal features from 3D ResNet
        feats = self.visual_backbone(video_frames)  # [B, T', 512]
        
        # 2) Map 512 -> d_model
        feats = self.visual_fc(feats)  # [B, T', d_model]
        
        # 3) Transpose to [T', B, d_model] for Transformer
        feats = feats.transpose(0, 1)  # [T', B, d_model]
        
        # 4) Add positional encoding
        feats = self.pos_encoder_vis(feats)  # [T', B, d_model]
        
        # 5) Pass into Transformer encoder
        # If you use nn.Transformer directly, we can do:
        memory = self.transformer.encoder(feats)  # [T', B, d_model]
        return memory

    def decode_text(self, tgt_tokens, memory):
        """
        tgt_tokens: [B, L] token IDs (decoder input)
        memory: [T', B, d_model] from the encoder
        Returns logits: [B, L, vocab_size]
        """
        # 1) Embed target tokens
        emb = self.token_embedding(tgt_tokens)  # [B, L, d_model]
        
        # 2) Transpose to [L, B, d_model] for Transformer
        emb = emb.transpose(0, 1)
        
        # 3) Positional encoding for the decoder
        emb = self.pos_encoder_txt(emb)  # [L, B, d_model]
        
        # 4) Transformer decoder
        # We need a causal mask so the model doesn't peek ahead
        L = tgt_tokens.size(1)
        tgt_mask = self.generate_subsequent_mask(L).to(tgt_tokens.device)
        
        # Optionally, you can also incorporate source/target padding masks
        decoded = self.transformer.decoder(emb, memory, tgt_mask=tgt_mask)
        
        # 5) Project to vocab
        decoded = decoded.transpose(0, 1)  # [B, L, d_model]
        logits = self.output_fc(decoded)   # [B, L, vocab_size]
        return logits

    def forward(self, video_frames, tgt_tokens):
        """
        video_frames: [B, C, T, H, W] (the lip-cropped clip)
        tgt_tokens: [B, L] (the input token IDs for teacher-forcing)
        Returns: [B, L, vocab_size] (logits over the vocabulary)
        """
        # Encode the video
        memory = self.encode_video(video_frames)  # [T', B, d_model]
        
        # Decode the text
        logits = self.decode_text(tgt_tokens, memory)  # [B, L, vocab_size]
        return logits

    def generate_subsequent_mask(self, size):
        """
        Generates an upper-triangular matrix of -inf, used for masking out 
        future tokens in the decoder.
        shape: [size, size]
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask
