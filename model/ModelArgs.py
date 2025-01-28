from dataclasses import dataclass

from model.helper import getDevice


@dataclass
class TransformerArgs:
    vocab_size: int
    d_model: int
    d_ff: int
    n_head: int
    num_latents: int
    num_layers: int
    dropout: float
    device = getDevice()
    max_len: int = 1000
