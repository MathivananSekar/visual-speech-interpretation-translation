import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18


class SpatioTemporalResNet(nn.Module):
    """
    Uses torchvision's 3D ResNet (r3d_18) as a backbone.
    We'll remove the final FC layer so we can extract spatiotemporal features.
    By default, r3d_18 outputs [B, 512] after pooling. 
    We will adapt it to keep a sequence dimension if possible.
    """
    def __init__(self, return_sequence=True):
        super(SpatioTemporalResNet, self).__init__()
        
        # Load a 3D ResNet-18 model
        base_model = r3d_18(pretrained=False)
        
        # Extract components from the base model
        # This includes conv1, bn1, relu, maxpool
        self.stem = base_model.stem

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # We can keep the final pooling or adapt it 
        # depending on whether we want a single vector or a sequence.
        # If return_sequence=False, we do global spatiotemporal pooling -> [B, 512]
        # If return_sequence=True, we might keep temporal dimension, 
        # but that requires custom modifications. 
        # For simplicity, let's do global *spatial* pool but keep time dimension.
        
        self.return_sequence = return_sequence
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))  
        # (None, 1, 1) means we keep the temporal dimension but pool over H and W

    def forward(self, x):
        """
        x shape expected: [B, C, T, H, W]
        returns:
          if return_sequence=True -> [B, feat_dim, T] (i.e. a feature for each time step)
          if return_sequence=False -> [B, feat_dim]
        """
        # Pass through 3D ResNet layers
        x = self.stem(x)      # [B, 64, T/2, H/2, W/2] typically
        x = self.layer1(x)    # e.g. [B, 64, T/2, H/2, W/2]
        x = self.layer2(x)    # e.g. [B, 128, T/4, H/4, W/4]
        x = self.layer3(x)    # e.g. [B, 256, T/8, H/8, W/8]
        x = self.layer4(x)    # e.g. [B, 512, T/8, H/8, W/8]
        
        # Pool over spatial dims, keep time
        x = self.avgpool(x)   # shape [B, 512, T/8, 1, 1]
        
        # Now we have [B, 512, T'], where T' ~ T/8 
        # (depends on how the conv blocks subsample the time dimension)
        if self.return_sequence:
            # Flatten to [B, 512, T']
            x = x.squeeze(-1).squeeze(-1)  # remove H,W dims -> [B, 512, T']
            # Transpose to [B, T', 512]
            x = x.transpose(1, 2)  # [B, T', 512]
        else:
            # Global pool over time as well
            x = F.adaptive_avg_pool1d(x.squeeze(-1).squeeze(-1), 1)  # [B, 512, 1]
            x = x.squeeze(-1)  # [B, 512]
        
        return x
