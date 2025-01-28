import torch

from ..ModelArgs import TransformerArgs
from ..Block import DecoderBlock
from ..helper import getDevice


def test_DecoderBlock():
    model_args = TransformerArgs(
        d_model=64,
        d_ff=256,
        n_head=8,
        num_latents=16,
        num_layers=4,
        dropout=0.1,
        device=getDevice(),
    )

    model = DecoderBlock(
        d_model=model_args.d_model,
        d_ff=model_args.d_ff,
        n_head=model_args.n_head,
        num_latents=model_args.num_latents,
        dropout=model_args.dropout,
    ).to(model_args.device)

    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, model_args.d_model).to(model_args.device)
    mask = torch.randn(batch_size, 1, seq_len, seq_len + model_args.num_latents).to(
        model_args.device
    )

    output = model(x, mask)

    assert output.shape == (batch_size, seq_len, model_args.d_model), (
        f"Expected output shape {(batch_size, seq_len, model_args.d_model)}, got {output.shape}"
    )

    print("All tests passed!")


if __name__ == "__main__":
    test_DecoderBlock()
