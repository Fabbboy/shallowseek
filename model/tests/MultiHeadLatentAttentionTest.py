import torch
from ..MultiHeadLatentAttention import MultiHeadLatentAttention
from ..ModelArgs import TransformerArgs
from ..helper import causal_mask, getDevice


def test_MultiHeadLatentAttention():
    model_args = TransformerArgs(
        vocab_size=1000,  # Not used here, but required by TransformerArgs
        d_model=64,
        d_ff=256,  # Not used here, but part of TransformerArgs
        n_head=8,
        num_latents=16,
        num_layers=4,  # Not used here, but part of TransformerArgs
        dropout=0.1,
    )

    device = getDevice()
    model = MultiHeadLatentAttention(
        d_model=model_args.d_model,
        num_heads=model_args.n_head,
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
        model_args.n_head,
        seq_len,
        seq_len + model_args.num_latents,
    ), (
        f"Expected attn_probs shape {(batch_size, model_args.n_head, seq_len, seq_len + model_args.num_latents)}, "
        f"got {attn_probs.shape}"
    )

    print("All tests passed!")


if __name__ == "__main__":
    test_MultiHeadLatentAttention()
