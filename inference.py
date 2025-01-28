import os
from process.Dataset import SequenceDataset
from tqdm import tqdm
from torch.cuda.amp import autocast
from process.Tokenizer import BPETokenizer
from datasets import load_dataset
from model.helper import causal_mask
from model.ModelArgs import TransformerArgs
from model.Transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
import accelerate
from transformers import get_linear_schedule_with_warmup

acc = accelerate.Accelerator()
device = acc.device

model_args = TransformerArgs(
    vocab_size=15000,
    d_model=512,
    d_ff=2048,
    n_head=8,
    num_latents=16,
    num_layers=4,
    dropout=0.1,
)

def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    context = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = context

    with torch.no_grad():
        for _ in range(max_length):
            mask = causal_mask(generated.size(1), model_args.num_latents, device=device)
            output = model(generated, mask)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0].tolist())

# Load the saved model to get the correct max_len
saved_model = torch.load("model.pth", map_location=device)
max_len = saved_model['pos_enc.pe'].shape[1]

model = Transformer(
    model_args.vocab_size,
    model_args.d_model,
    model_args.d_ff,
    model_args.n_head,
    model_args.num_latents,
    model_args.num_layers,
    model_args.dropout,
    max_len=max_len  # Use the max_len from the saved model
).to(device)

model.load_state_dict(saved_model)
tokenizer = BPETokenizer(model_args.vocab_size, min_frequency=2)
tokenizer.load("tokenizer.json")

print("Generating sample text:")
prompt = "The quick brown fox"
generated_text = generate_text(model, tokenizer, prompt, max_length=50)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")

