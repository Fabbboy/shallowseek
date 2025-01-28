import torch
from Embedding import Embedding

def test_Embedding():
    vocab_size = 1000
    d_model = 512
    batch_size = 4
    seq_len = 20

    embedding_layer = Embedding(vocab_size, d_model)

    assert embedding_layer.embedding.weight.size() == (vocab_size, d_model), \
        f"Expected embedding weight shape ({vocab_size}, {d_model}), got {embedding_layer.embedding.weight.size()}"

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # Random token IDs
    output = embedding_layer(input_ids)
    assert output.size() == (batch_size, seq_len, d_model), \
        f"Expected output shape ({batch_size}, {seq_len}, {d_model}), got {output.size()}"

    raw_embeddings = embedding_layer.embedding(input_ids)
    scaled_embeddings = raw_embeddings * (d_model**0.5)
    assert torch.allclose(output, scaled_embeddings, atol=1e-5), \
        "The output is not correctly scaled by sqrt(d_model)"

    print("All tests passed!")


if __name__ == "__main__":
    test_Embedding()
