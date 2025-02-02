import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class LipReadingModel(nn.Module):
    def __init__(self, num_classes, vocab_size, hidden_dim=256, nhead=4, num_encoder_layers=4, num_decoder_layers=2):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # 1) Pretrained ResNet for feature extraction (2D)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # remove classification layer -> output size 512

        # Project to desired hidden_dim
        self.linear_in = nn.Linear(512, hidden_dim)

        # 2) Transformer Encoder (temporal)
        enc_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # 3) CTC Head
        self.ctc_fc = nn.Linear(hidden_dim, num_classes)  # output for CTC

        # 4) Attention-based Decoder
        dec_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.decoder = TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attn_fc = nn.Linear(hidden_dim, vocab_size)

    def forward_encoder(self, x):
        """
        x: (B, T, 3, 64, 128)
        Returns encoder_out: (T, B, hidden_dim)
        """
        B, T, C, H, W = x.shape

        # (a) Flatten so we can pass each frame through ResNet
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)   # (B*T, 512)
        feats = self.linear_in(feats)  # (B*T, hidden_dim)

        # (b) Reshape to (B, T, hidden_dim), then permute for transformer
        feats = feats.view(B, T, self.hidden_dim)  # (B, T, hidden_dim)
        feats = feats.permute(1, 0, 2)  # (T, B, hidden_dim)

        # (c) Pass through transformer encoder
        memory = self.transformer_encoder(feats)  # (T, B, hidden_dim)
        return memory

    def forward_ctc(self, encoder_out):
        """
        encoder_out: (T, B, hidden_dim)
        -> ctc_logits: (B, T, num_classes)
        """
        x = encoder_out.permute(1, 0, 2)  # (B, T, hidden_dim)
        return self.ctc_fc(x)            # (B, T, num_classes)

    def forward_decoder(self, encoder_out, tgt_tokens):
        """
        encoder_out: (T, B, hidden_dim)
        tgt_tokens:  (B, U)
        Returns: (B, U, vocab_size)
        """
        B, U = tgt_tokens.shape
        # embed target
        tgt_emb = self.embedding(tgt_tokens)  # (B, U, hidden_dim)
        tgt_emb = tgt_emb.permute(1, 0, 2)    # (U, B, hidden_dim)

        # decode
        dec_out = self.decoder(tgt_emb, encoder_out)  # (U, B, hidden_dim)

        # project to vocab
        dec_out = dec_out.permute(1, 0, 2)    # (B, U, hidden_dim)
        logits = self.attn_fc(dec_out)        # (B, U, vocab_size)
        return logits

    def forward(self, x, tgt_tokens=None):
        # x: (B, T, 3, 64, 128)
        # tgt_tokens: (B, U) or None
        encoder_out = self.forward_encoder(x)
        ctc_logits  = self.forward_ctc(encoder_out)
        attn_logits = None
        if tgt_tokens is not None:
            attn_logits = self.forward_decoder(encoder_out, tgt_tokens)
        return ctc_logits, attn_logits
