import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.d_model**0.5
