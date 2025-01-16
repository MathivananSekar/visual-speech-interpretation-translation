import torch
import torch.nn as nn
import torch.nn.functional as F

class LipReadingCNN(nn.Module):
    def __init__(self):
        super(LipReadingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 112 * 112, 256)
        self.fc2 = nn.Linear(256, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 64 * 112 * 112)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x