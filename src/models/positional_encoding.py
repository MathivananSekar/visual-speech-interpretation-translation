import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds a (sine-based) positional encoding to the input features.
    This helps the Transformer learn positional information in a sequence.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe -> [max_len, d_model], but Transformer expects [seq_len, batch_size, d_model]
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        # Add positional encoding
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)
