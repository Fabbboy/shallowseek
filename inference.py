import torch
import torch.nn.functional as F
from process.Tokenizer import BPETokenizer
from model.Transformer import Transformer
from model.ModelArgs import TransformerArgs
import accelerate
from model.helper import causal_mask

acc = accelerate.Accelerator()
device = acc.device

model_args = TransformerArgs(
    vocab_size=15000,
    d_model=256,
    d_ff=1024,
    n_head=8,
    num_latents=16,
    num_layers=4,
    dropout=0.1,
)


def generate_text(
    model, tokenizer: BPETokenizer, prompt, max_length=50, temperature=1.0, top_k=50
):
    model.eval()
    context = (
        torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    )  # Adding batch dimension
    generated = context

    print(f"Initial generated shape: {generated.shape}")

    with torch.no_grad():
        for _ in range(
            max_length // 4
        ):  # Divide by 4 since we're generating 4 tokens at a time
            mask = causal_mask(generated.size(1), model_args.num_latents, device=device)
            print(f"Mask shape: {mask.shape}")

            output = model(generated, mask)
            print(f"Model output shape: {output.shape}")

            next_token_logits = output[:, -4:, :]  # Take the last 4 token predictions
            print(f"Next token logits shape: {next_token_logits.shape}")

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            next_tokens = []
            for i in range(4):
                # Apply top-k filtering for each token
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits[:, i, :], top_k)
                    probabilities = F.softmax(values, dim=-1)
                    next_token = torch.multinomial(probabilities, 1)
                    next_token = indices.gather(-1, next_token)
                else:
                    probabilities = F.softmax(next_token_logits[:, i, :], dim=-1)
                    next_token = torch.multinomial(probabilities, 1)

                next_tokens.append(next_token)

            next_tokens = torch.cat(next_tokens, dim=1)
            print(f"Next tokens shape: {next_tokens.shape}")

            # Concatenate generated tokens
            generated = torch.cat((generated, next_tokens), dim=1)
            print(f"Updated generated shape: {generated.shape}")

            if tokenizer.get_eos_token() in next_tokens[0].tolist():
                print("EOS token found, stopping generation")
                break

    return tokenizer.decode(generated[0].tolist())


# Load the saved model to get the correct max_len
saved_model = torch.load("model.pth", map_location=device)
max_len = saved_model["pos_enc.pe"].shape[1]

model = Transformer(
    model_args.vocab_size,
    model_args.d_model,
    model_args.d_ff,
    model_args.n_head,
    model_args.num_latents,
    model_args.num_layers,
    model_args.dropout,
    max_len=max_len,  # Use the max_len from the saved model
).to(device)

model.load_state_dict(saved_model)
tokenizer = BPETokenizer(model_args.vocab_size, min_frequency=2)
tokenizer.load("tokenizer.json")

# Example usage with different sampling strategies
prompt = "The quick brown fox"
generated_text = generate_text(
    model, tokenizer, prompt, max_length=50, temperature=0.6, top_k=50
)

print("Prompt:", prompt)
print("Generated Text:", generated_text)
