from dataclasses import dataclass
import torch
from .Block import DecoderBlock
from .helper import getDevice


@dataclass
class ModelArgs:
    d_model: int
    d_ff: int
    n_head: int
    num_latents: int
    dropout: float
    device = getDevice()


def test_DecoderBlock():
    model_args = ModelArgs(d_model=64, d_ff=128, n_head=8, num_latents=16, dropout=0.1)

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
