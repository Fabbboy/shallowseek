import torch
from MultiHeadLatentAttention import MultiHeadLatentAttention
from dataclasses import dataclass

from helper import causal_mask, getDevice


@dataclass
class ModelArgs:
    """
    Data class for storing model configuration parameters.

    Attributes:
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        num_latents (int): Number of latent vectors.
        dropout (float): Dropout probability.
    """

    d_model: int
    num_heads: int
    num_latents: int
    dropout: float = 0.1


def test_MultiHeadLatentAttention():
    model_args = ModelArgs(
        d_model=64,
        num_heads=8,
        num_latents=16,
        dropout=0.1,
    )

    device = getDevice()
    model = MultiHeadLatentAttention(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        num_latents=model_args.num_latents,
        dropout=model_args.dropout,
    ).to(device)

    batch_size = 4
    seq_len = 32
    query = torch.randn(batch_size, seq_len, model_args.d_model).to(device)
    key = torch.randn(batch_size, seq_len, model_args.d_model).to(device)
    value = torch.randn(batch_size, seq_len, model_args.d_model).to(device)

    mask = causal_mask(seq_len, model_args.num_latents, device=device)

    output, attn_probs = model(query, key, value, mask)

    assert output.shape == (batch_size, seq_len, model_args.d_model), (
        f"Expected output shape {(batch_size, seq_len, model_args.d_model)}, got {output.shape}"
    )

    assert attn_probs.shape == (
        batch_size,
        model_args.num_heads,
        seq_len,
        seq_len + model_args.num_latents,
    ), (
        f"Expected attn_probs shape {(batch_size, model_args.num_heads, seq_len, seq_len + model_args.num_latents)}, "
        f"got {attn_probs.shape}"
    )

    print("All tests passed!")


if __name__ == "__main__":
    test_MultiHeadLatentAttention()
