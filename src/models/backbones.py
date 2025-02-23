import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights

class SpatioTemporalResNet(nn.Module):
    def __init__(self, return_sequence=True):
        super().__init__()
        base_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.stem = base_model.stem
        self.stem[0].stride = (1, 2, 2)  # Reduce temporal stride from 2 to 1
        self.layer1 = base_model.layer1
        # Ensure layer1 preserves temporal dimension
        for block in self.layer1:
            block.conv1[0].stride = (1, 1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1, 1)  # Match residual path
        
        self.layer2 = base_model.layer2
        # Adjust layer2 to downsample T once (e.g., 75 → 38)
        for block in self.layer2:
            block.conv1[0].stride = (1, 2, 2)  # First block downsamples H, W
            if block.downsample is not None:
                block.downsample[0].stride = (1, 2, 2)  # Match residual path
            break  # Only first block downsamples
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.return_sequence = return_sequence
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        assert x.dim() == 5 and x.shape[1] == 3, f"Expected [B, 3, T, H, W], got {x.shape}"
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.return_sequence:
            x = x.squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, T', 512]
        else:
            x = F.adaptive_avg_pool1d(x.squeeze(-1).squeeze(-1), 1).squeeze(-1)  # [B, 512]
        return x
