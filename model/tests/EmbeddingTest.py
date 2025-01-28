import torch
from ..Embedding import Embedding
from ..ModelArgs import TransformerArgs


def test_Embedding():
    model_args = TransformerArgs(
        vocab_size=1000,
        d_model=64,
        d_ff=256,
        n_head=8,
        num_latents=16,
        num_layers=4,
        dropout=0.1,
    )
    embedding_layer = Embedding(
        vocab_size=model_args.vocab_size, d_model=model_args.d_model
    ).to(model_args.device)

    assert embedding_layer.embedding.weight.size() == (
        model_args.vocab_size,
        model_args.d_model,
    ), (
        f"Expected embedding weight shape ({model_args.vocab_size}, {model_args.d_model}), "
        f"got {embedding_layer.embedding.weight.size()}"
    )

    input_ids = torch.randint(
        0, model_args.vocab_size, (4, 32), device=model_args.device
    )

    output = embedding_layer(input_ids)

    assert output.size() == (4, 32, 64), (
        f"Expected output shape (4, 32, 64), got {output.size()}"
    )

    raw_embeddings = embedding_layer.embedding(input_ids)
    scaled_embeddings = raw_embeddings * (model_args.d_model**0.5)
    assert torch.allclose(output, scaled_embeddings, atol=1e-5), (
        "The output is not correctly scaled by sqrt(d_model)"
    )

    print("All tests passed!")


if __name__ == "__main__":
    test_Embedding()
