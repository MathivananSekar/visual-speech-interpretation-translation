import torch
import torch.nn as nn

class VSRModel(nn.Module):
    def __init__(self, n_mfcc=160, num_classes=39):
        super(VSRModel, self).__init__()
        
        # Video Encoder
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),   # shape => [B, 16, depth, H/2, W/2]
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),   # shape => [B, 32, depth/2, H/4, W/4]
            #
            # Additional step to collapse depth => 1
            # shape => [B, 32, 1, 56, 56]
            #
            nn.AdaptiveAvgPool3d((1, 56, 56))
        )
        
        # Audio Encoder
        self.audio_encoder = nn.LSTM(input_size=n_mfcc, hidden_size=128, num_layers=2, batch_first=True)
        
        # Fusion Layer
        # After AdaptiveAvgPool3d, shape is [batch_size, 32, 1, 56, 56]
        # => Flatten to [batch_size, 32 * 56 * 56] = 100352
        # Combined with audio_features=128 => 100480
        self.fusion = nn.Linear(100352 + 128, 256)
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, video, audio):
        """
        video shape: [batch_size, 3, depth, height, width]
        audio shape: [batch_size, time_steps, n_mfcc]
        """
        # Video Encoder
        # e.g., if video has shape [B, 3, 75, 224, 224],
        # after the last pooling, shape => [B, 32, 1, 56, 56]
        video_out = self.video_encoder(video)
        
        # Flatten => [B, 32 * 56 * 56] = [B, 100352]
        video_features = video_out.flatten(start_dim=1)
        
        # Audio Encoder => LSTM
        # shape: [batch_size, time_steps, input_size]
        audio_out, _ = self.audio_encoder(audio)
        
        # Take the last time-step from LSTM => shape: [batch_size, 128]
        audio_features = audio_out[:, -1, :]

        # Fusion => cat => [batch_size, 100352 + 128] = [batch_size, 100480]
        combined_features = torch.cat((video_features, audio_features), dim=1)
        
        # Pass through fusion => [batch_size, 256]
        fused = self.fusion(combined_features)
        
        # Classification => [batch_size, 39]
        output = self.classifier(fused)

        return output
