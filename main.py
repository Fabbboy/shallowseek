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


def load_dat() -> tuple[list[str], list[str]]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_data = dataset["train"]["text"]
    valid = dataset["validation"]["text"]

    # Use only 25% of the dataset
    train_data = train_data[: len(train_data) // 4]
    valid = valid[: len(valid) // 4]

    return train_data, valid


train_data, valid = load_dat()

model_args = TransformerArgs(
    vocab_size=15000,
    d_model=256,
    d_ff=1024,
    n_head=8,
    num_latents=16,
    num_layers=4,
    dropout=0.1,
)

accelerator = accelerate.Accelerator()
device = accelerator.device

# Tokenize the dataset
tokenizer_path = "tokenizer.json"
tokenizer = BPETokenizer(model_args.vocab_size, min_frequency=2)
if os.path.exists(tokenizer_path):
    tokenizer.load(tokenizer_path)
else:
    tokenizer.train(train_data)
    tokenizer.save(tokenizer_path)

train_data = [
    encoding.ids for encoding in tokenizer.inner_tokenizer().encode_batch(train_data)
]
valid = [encoding.ids for encoding in tokenizer.inner_tokenizer().encode_batch(valid)]

BATCH_SIZE = 64
CONTEXT_LEN = 256
TARGET_LEN = 4
EPOCHS = 10


def collate_fn(batch):
    contexts, targets = zip(*batch)
    contexts = torch.nn.utils.rnn.pad_sequence(
        contexts, batch_first=True, padding_value=0
    )
    targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )
    return contexts, targets

eos = tokenizer.get_eos_token()
print(f"EOS token: {eos}")
print(type(eos))

train_dataset = SequenceDataset(
    CONTEXT_LEN,
    TARGET_LEN,
    train_data,
    tokenizer.get_pad_token(),
    tokenizer.get_eos_token(),
    verbose=True,
)
valid_dataset = SequenceDataset(
    CONTEXT_LEN,
    TARGET_LEN,
    valid,
    tokenizer.get_pad_token(),
    tokenizer.get_eos_token(),
    verbose=True,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# Create the model
model = Transformer(
    vocab_size=model_args.vocab_size,
    d_model=model_args.d_model,
    d_ff=model_args.d_ff,
    n_head=model_args.n_head,
    num_latents=model_args.num_latents,
    num_layers=model_args.num_layers,
    dropout=model_args.dropout,
    max_len=CONTEXT_LEN,
).to(device)

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters (Human): {parameters:,}")
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
pad = tokenizer.get_pad_token()
print(f"Pad token: {pad}")
print(type(pad))
criterion = nn.CrossEntropyLoss(ignore_index=pad)

model, optimizer, train_loader, valid_loader, criterion = accelerator.prepare(
    model, optimizer, train_loader, valid_loader, criterion
)

# Learning rate scheduler with warmup
num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = num_training_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)


def calculate_loss(output, target, criterion):
    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    non_pad_mask = target.ne(tokenizer.get_pad_token())
    num_tokens = non_pad_mask.sum().item()
    return loss.sum() / num_tokens if num_tokens > 0 else loss.sum()


def log_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"{name}: grad_norm: {param.grad.norm().item()}")


def train_step(batch):
    context, target = batch
    context, target = context.to(device), target.to(device)

    optimizer.zero_grad()
    mask = causal_mask(CONTEXT_LEN, model_args.num_latents, device=device)

    output = model(context, mask)
    output = output[:, -TARGET_LEN:, :]
    output = output.reshape(-1, model_args.vocab_size)
    target = target.reshape(-1)

    loss = calculate_loss(output, target, criterion)

    accelerator.backward(loss)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    return loss.item()


def eval_step(batch):
    context, target = batch
    context, target = context.to(device), target.to(device)

    mask = causal_mask(CONTEXT_LEN, model_args.num_latents, device=device)
    output = model(context, mask)

    output = output[:, -TARGET_LEN:, :]
    output = output.reshape(-1, model_args.vocab_size)
    target = target.reshape(-1)

    loss = calculate_loss(output, target, criterion)
    return loss.item()


model.train()
last_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    train_bar = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Training]", leave=False
    )
    for batch in train_bar:
        loss = train_step(batch)
        total_loss += loss
        num_batches += 1
        train_bar.set_postfix({"Loss": loss})

    avg_train_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")

    # Log gradients after each epoch
    log_gradients(model)

    model.eval()
    total_loss = 0.0
    num_batches = 0
    valid_bar = tqdm(
        valid_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Validation]", leave=False
    )

    with torch.no_grad():
        for batch in valid_bar:
            loss = eval_step(batch)
            total_loss += loss
            num_batches += 1
            valid_bar.set_postfix({"Loss": loss})

    avg_val_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < last_loss:
        torch.save(model.state_dict(), "model.pth")
        last_loss = avg_val_loss
