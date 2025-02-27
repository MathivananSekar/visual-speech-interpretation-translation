import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class SpatioTemporalResNet(nn.Module):
    def __init__(self, return_sequence=True):
        super().__init__()
        base_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

        self.stem = base_model.stem
        self.stem[0].stride = (1, 2, 2)

        self.layer1 = base_model.layer1
        # Ensure layer1 preserves temporal dimension
        for block in self.layer1:
            block.conv1[0].stride = (1, 1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1, 1)
        
        self.layer2 = base_model.layer2
        # Adjust layer2 to downsample T once
        first_block = True
        for block in self.layer2:
            if first_block:
                block.conv1[0].stride = (1, 2, 2)
                if block.downsample is not None:
                    block.downsample[0].stride = (1, 2, 2)
                first_block = False

        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.return_sequence = return_sequence
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        # x shape: [B, 3, T, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # shape [B, C, T', 1, 1]

        if self.return_sequence:
            # Squeeze spatial dims and transpose so shape is [B, T', C]
            x = x.squeeze(-1).squeeze(-1).transpose(1, 2)
        else:
            # Single global feature per clip
            x = F.adaptive_avg_pool1d(x.squeeze(-1).squeeze(-1), 1).squeeze(-1)

        return x
