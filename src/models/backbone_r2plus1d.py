import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class SpatioTemporalR2Plus1D(nn.Module):
    """
    A larger video backbone using R(2+1)D-18 from torchvision.models.video.
    We'll expose final feature dim=512 if return_sequence=True
    (temporal dimension remains).
    """
    def __init__(self, return_sequence=True):
        super().__init__()
        # Load pretrained on Kinetics-400
        base_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        self.stem = base_model.stem
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Usually final pooling is adaptive avgpool -> (1,1,1)
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.return_sequence = return_sequence

    def forward(self, x):
        # x: [B, 3, T, H, W]
        x = self.stem(x)        # -> [B, 64, T/1, H/2, W/2]
        x = self.layer1(x)      # -> [B, 64, T/1, H/2, W/2]
        x = self.layer2(x)      # -> [B, 128, T/2, H/4, W/4]
        x = self.layer3(x)      # -> [B, 256, T/4, H/8, W/8]
        x = self.layer4(x)      # -> [B, 512, T/8, H/16, W/16]

        # We do an AdaptiveAvgPool3d to keep only T dimension
        x = self.avgpool(x)     # -> [B, 512, T/8, 1, 1]

        if self.return_sequence:
            # Squeeze spatial dims
            # shape -> [B, 512, T']
            x = x.squeeze(-1).squeeze(-1)
            # shape -> [B, T', 512]
            x = x.transpose(1, 2)
        else:
            # single global feature
            x = F.adaptive_avg_pool1d(x.squeeze(-1).squeeze(-1), 1).squeeze(-1)
        return x
