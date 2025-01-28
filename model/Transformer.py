import torch
import torch.nn as nn
from .Block import DecoderBlock

class Transformer(nn.Module):
    """
    Transformer model with latent attention mechanism.

    Args:
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feedforward network.
        n_head (int): Number of attention heads.
        num_latents (int): Number of latent vectors for latent attention.
        num_layers (int): Number of decoder blocks.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
        num_latents: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(d_model, d_ff, n_head, num_latents, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len + num_latents).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.decoder:
            x = layer(x, mask)

        return x
