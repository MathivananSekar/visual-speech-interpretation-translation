# backbones_3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights

class VideoEncoder3D(nn.Module):
    """
    3D CNN that preserves the time dimension, returning [B, T_v, output_dim].
    """
    def __init__(self, output_dim=256):
        super().__init__()
        base_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.stem = base_model.stem
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.final_fc = nn.Linear(512, output_dim)

    def forward(self, x):
        # x => [B, 3, T, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # spatial pool => [B, 512, T', 1, 1]
        x = self.avgpool(x)
        # => [B, 512, T']
        x = x.squeeze(-1).squeeze(-1).transpose(1, 2)
        x = self.final_fc(x)  # => [B, T', output_dim]
        return x
