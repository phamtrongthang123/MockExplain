"""An intentionally uncovered model family: no generic method list is invented."""
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(12, 8)
        self.output = nn.Linear(8, 2)

    def forward(self, x):
        return self.output(self.hidden(x).relu())
