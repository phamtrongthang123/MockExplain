"""Small representative language transformer; not a pretrained model."""
import torch
from torch import nn


class TinyLanguageTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(64, 16)
        self.position_embedding = nn.Embedding(128, 16)
        self.block = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 64, bias=False)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool), diagonal=1)
        return self.lm_head(self.norm(self.block(x, src_mask=mask)))
