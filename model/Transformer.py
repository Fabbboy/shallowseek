import torch
import torch.nn as nn
from .Block import DecoderBlock
from .PositionalEncoding import PositionalEncoding
from .Embedding import Embedding


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_head: int,
        num_latents: int,
        num_layers: int,
        dropout: float,
        max_len: int = 1000,
    ):
        super(Transformer, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, d_ff, n_head, num_latents, dropout)
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(d_model, vocab_size)  # Projection layer

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.proj(x)
        return x
