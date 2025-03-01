import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class LipReadingModel(nn.Module):
    def __init__(self, num_classes, vocab_size, hidden_dim=256, nhead=4, num_encoder_layers=4, num_decoder_layers=2, return_sequence=False):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.return_sequence = return_sequence

        # 1) Pretrained ResNet for feature extraction (2D)
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Remove classification layer -> output size 512

        # Project to desired hidden_dim
        self.linear_in = nn.Linear(512, hidden_dim)

        # 2) Transformer Encoder (temporal)
        enc_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # 3) CTC Head
        self.ctc_fc = nn.Linear(hidden_dim, num_classes)  # Output for CTC

        # 4) Attention-based Decoder
        dec_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.decoder = TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attn_fc = nn.Linear(hidden_dim, vocab_size)

    def forward_encoder(self, x):
        """
        x: (B, T, 3, 64, 128)
        Returns encoder_out: (B, T, hidden_dim)
        """
        B, T, C, H, W = x.shape

        # (a) Flatten so we can pass each frame through ResNet
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)  # (B*T, 512)
        feats = self.linear_in(feats)  # (B*T, hidden_dim)

        # (b) Reshape to (B, T, hidden_dim)
        feats = feats.view(B, T, self.hidden_dim)  # (B, T, hidden_dim)

        # (c) Pass through transformer encoder
        memory = self.transformer_encoder(feats)  # (B, T, hidden_dim)
        return memory

    def forward_ctc(self, encoder_out):
        """
        encoder_out: (B, T, hidden_dim)
        -> ctc_logits: (B, T, num_classes)
        """
        return self.ctc_fc(encoder_out)  # (B, T, num_classes)

    def forward_decoder(self, encoder_out, tgt_tokens):
        """
        encoder_out: (B, T, hidden_dim)
        tgt_tokens:  (B, U)
        Returns: (B, U, vocab_size)
        """
        # Embed target
        tgt_emb = self.embedding(tgt_tokens)  # (B, U, hidden_dim)

        # Decode
        dec_out = self.decoder(tgt_emb, encoder_out)  # (B, U, hidden_dim)

        # Project to vocab
        logits = self.attn_fc(dec_out)  # (B, U, vocab_size)
        return logits

    def forward(self, x, tgt_tokens=None):
        """
        x: (B, T, 3, 64, 128)
        tgt_tokens: (B, U) or None
        """
        encoder_out = self.forward_encoder(x)  # (B, T, hidden_dim)
        
        if self.return_sequence:
            return encoder_out  # Return features for backbone use
        
        ctc_logits = self.forward_ctc(encoder_out)  # (B, T, num_classes)
        attn_logits = None
        if tgt_tokens is not None:
            attn_logits = self.forward_decoder(encoder_out, tgt_tokens)  # (B, U, vocab_size)
        return ctc_logits, attn_logits