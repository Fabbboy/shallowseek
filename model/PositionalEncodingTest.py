from dataclasses import dataclass
import torch
from PositionalEncoding import PositionalEnconding


@dataclass
class ModelArgs:
    d_model: int
    max_len: int


def test_PositionalEnconding():
    model_args = ModelArgs(d_model=512, max_len=5000)
    model = PositionalEnconding(model_args.d_model, model_args.max_len)

    assert model.pe.size() == (1, model_args.max_len, model_args.d_model), (
        f"Expected shape (1, {model_args.max_len}, {model_args.d_model}), got {model.pe.size()}"
    )

    batch_size = 2
    seq_len = 100
    x = torch.randn(
        batch_size, seq_len, model_args.d_model
    )
    output = model(x)

    assert output.size() == x.size(), (
        f"Expected output shape {x.size()}, got {output.size()}"
    )

    pe_slice = model.pe[:, :seq_len, :]
    assert torch.allclose(output - x, pe_slice, atol=1e-5), (
        "Positional encoding was not added correctly to the input tensor"
    )

    print("All tests passed!")


if __name__ == "__main__":
    test_PositionalEnconding()
