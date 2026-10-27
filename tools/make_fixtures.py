"""Regenerate trusted test/example tensor archives. Requires optional CPU PyTorch.

This developer utility is never imported by the app. It constructs only the tiny
models below and never imports an uploaded source file or runs a forward pass.
Fixtures were generated with PyTorch 2.14.1+cpu; application/tests do not need it.
"""
from collections import OrderedDict
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.norm = nn.BatchNorm2d(8)
        self.head = nn.Linear(8, 4)


def main():
    torch.manual_seed(0)
    torch.save(CNN().state_dict(), ROOT / 'examples/tiny_cnn.pt')
    tensors = OrderedDict([
        ('bfloat', torch.tensor([0., 1., -2., 4.], dtype=torch.bfloat16)),
        ('bool', torch.tensor([True, False, True])),
        ('half', torch.tensor([0., 0.5, -1.], dtype=torch.float16)),
        ('int', torch.arange(5, dtype=torch.int64)),
        ('nonfinite', torch.tensor([float('inf'), float('nan'), 1.])),
    ])
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    tensors['view'] = base.t()
    tensors['alias'] = base
    tensors['empty'] = torch.empty(0, 2)
    tensors['offset'] = base[1:, 2:]
    for protocol in (2, 4, 5):
        torch.save(tensors, ROOT / f'tests/fixtures/tensors_p{protocol}.pt', pickle_protocol=protocol)
    torch.save({'state_dict': tensors, 'epoch': 2}, ROOT / 'tests/fixtures/wrapped.pt')
    torch.save({f'layer{i:02}.weight': torch.arange(1024, dtype=torch.float32) for i in range(40)}, ROOT / 'tests/fixtures/partial.pt')
    torch.save({'token_embedding.weight': torch.zeros(64, 16),
                'block.attn.q_proj.weight': torch.zeros(16, 16),
                'lm_head.weight': torch.zeros(64, 16)}, ROOT / 'tests/fixtures/language.pt')
    print(f'Generated fixtures with PyTorch {torch.__version__}; no model forward pass.')


if __name__ == '__main__':
    main()
