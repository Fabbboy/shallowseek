from dataclasses import dataclass
import torch
from .FeedForward import FeedForward


@dataclass
class ModelArgs:
    d_model: int
    d_ff: int
    dropout: float


def test_FeedForward():
    model_args = ModelArgs(d_model=512, d_ff=2048, dropout=0.1)
    model = FeedForward(model_args.d_model, model_args.d_ff, model_args.dropout)

    model.eval()

    assert model.linear1.weight.size() == (model_args.d_ff, model_args.d_model), (
        f"Expected linear1 weight shape ({model_args.d_ff}, {model_args.d_model}), "
        f"got {model.linear1.weight.size()}"
    )
    assert model.linear2.weight.size() == (model_args.d_model, model_args.d_ff), (
        f"Expected linear2 weight shape ({model_args.d_model}, {model_args.d_ff}), "
        f"got {model.linear2.weight.size()}"
    )

    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, model_args.d_model) 
    output = model(x)
    assert output.size() == x.size(), (
        f"Expected output shape {x.size()}, got {output.size()}"
    )

    raw_output = model.linear2(torch.relu(model.linear1(x)))
    assert torch.allclose(output, raw_output, atol=1e-5), (
        "The output does not match the expected FeedForward behavior"
    )

    print("All tests passed!")


if __name__ == "__main__":
    test_FeedForward()
