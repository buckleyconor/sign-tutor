"""Two-handed classifier for BSL (126 features → 26 classes)."""
import torch.nn as nn


class TwoHandClassifier(nn.Module):
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(126, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
