import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sine-based positional encoding.
    Supports both [seq_len, batch_size, d_model] (default) and 
    [batch_size, seq_len, d_model] (if batch_first=True).
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.batch_first = batch_first

        # Create constant 'pe' matrix with values dependent on position and i.
        pe = torch.zeros(max_len, d_model)  # shape: [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        # Register as buffer so that it's automatically moved to the right device.
        # For default (non-batch-first) inputs, expected shape is [seq_len, 1, d_model].
        if not batch_first:
            pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        else:
            # For batch_first inputs, we want [1, seq_len, d_model]
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor
               - If batch_first is False: [seq_len, batch_size, d_model]
               - If batch_first is True: [batch_size, seq_len, d_model]
        Returns:
            x with positional encoding added and dropout applied.
        """
        if not self.batch_first:
            seq_len = x.size(0)
            # Expand pe to match x's batch size, then add.
            x = x + self.pe[:seq_len].expand_as(x)
        else:
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len].expand_as(x)
        return self.dropout(x)
