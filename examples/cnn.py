"""Small representative CNN architecture. Upload this file; MockExplain only parses it."""
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        x = self.pool(self.norm(self.conv(x)).relu()).flatten(1)
        return self.head(x)
