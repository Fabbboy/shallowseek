import torch
from ..PositionalEncoding import PositionalEncoding
from ..ModelArgs import TransformerArgs


def test_PositionalEnconding():
    model_args = TransformerArgs(
        vocab_size=1000,  # Not used here, but required by TransformerArgs
        d_model=512,
        d_ff=2048,  # Not used here, but part of TransformerArgs
        n_head=8,  # Not used here, but part of TransformerArgs
        num_latents=16,  # Not used here, but part of TransformerArgs
        num_layers=4,  # Not used here, but part of TransformerArgs
        dropout=0.1,
        max_len=1000,
    )

    model = PositionalEncoding(model_args.d_model, model_args.max_len)
    assert model.pe.size() == (1, model_args.max_len, model_args.d_model), (
        f"Expected shape (1, {model_args.max_len}, {model_args.d_model}), got {model.pe.size()}"
    )

    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, model_args.d_model)
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
