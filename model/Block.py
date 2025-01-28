import torch
import torch.nn as nn
from .FeedForward import FeedForward
from .MultiHeadLatentAttention import MultiHeadLatentAttention


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block with latent attention and feedforward layers.

    Args:
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feedforward network.
        n_head (int): Number of attention heads.
        num_latents (int): Number of latent vectors for latent attention.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
        num_latents: int,
        dropout: float = 0.1,
    ):
        super(DecoderBlock, self).__init__()
        self.attn = MultiHeadLatentAttention(d_model, n_head, num_latents, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the DecoderBlock.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len + num_latents).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        attn_output, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x
