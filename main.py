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
    d_model=64,
    d_ff=256,
    n_head=8,
    num_latents=16,
    num_layers=4,
    dropout=0.1,
)

accelerator = accelerate.Accelerator(mixed_precision="no")
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
CONTEXT_LEN = 128
TARGET_LEN = 4
EPOCHS = 10


def collate_fn(batch):
    """
    Collate function for DataLoader to handle padding.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): Batch of (context, target) pairs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batched context and target tensors.
    """
    contexts, targets = zip(*batch)

    # Pad sequences to the same length
    contexts = torch.nn.utils.rnn.pad_sequence(
        contexts, batch_first=True, padding_value=0
    )
    targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )

    return contexts, targets


train_dataset = SequenceDataset(
    CONTEXT_LEN, TARGET_LEN, train_data, tokenizer.get_pad_token(), verbose=True
)
valid_dataset = SequenceDataset(
    CONTEXT_LEN, TARGET_LEN, valid, tokenizer.get_pad_token(), verbose=True
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
    max_len=CONTEXT_LEN + TARGET_LEN,
).to(device)

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters (Human): {parameters:,}")
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
pad = tokenizer.get_pad_token()
criterion = nn.CrossEntropyLoss()

model, optimizer, train_loader, valid_loader, criterion = accelerator.prepare(
    model, optimizer, train_loader, valid_loader, criterion
)


def train_step(batch):
    context, target = batch
    context, target = context.to(device), target.to(device)

    optimizer.zero_grad()
    mask = causal_mask(CONTEXT_LEN, model_args.num_latents, device=device)


    output = model(context, mask)

    output = output[:, -TARGET_LEN:, :]

    output = output.reshape(-1, model_args.vocab_size)
    target = target.reshape(-1)

    loss = criterion(output, target)

    accelerator.backward(loss)
    optimizer.step()

    return loss.item()


def eval_step(batch):
    context, target = batch
    context, target = context.to(device), target.to(device)

    mask = causal_mask(CONTEXT_LEN, model_args.num_latents, device=device)
    output = model(context, mask)

    output = output[:, -TARGET_LEN:, :]

    output = output.reshape(-1, model_args.vocab_size)
    target = target.reshape(-1)

    loss = criterion(output, target)
    return loss.item()


model.train()
last_loss = float("inf")

for epoch in range(EPOCHS):
    total_loss = 0.0

    train_bar = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Training]", leave=False
    )
    for batch in train_bar:
        loss = train_step(batch)
        total_loss += loss
        train_bar.set_postfix({"Loss": loss})

    print(f"Epoch {epoch + 1}, Total Training Loss: {total_loss}")

    model.eval()
    total_loss = 0.0
    valid_bar = tqdm(
        valid_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Validation]", leave=False
    )
    if total_loss < last_loss:
        torch.save(model.state_dict(), "model.pth")

    last_loss = total_loss
    for batch in valid_bar:
        loss = eval_step(batch)
        total_loss += loss
        valid_bar.set_postfix({"Loss": loss})

    print(f"Epoch {epoch + 1}, Total Validation Loss: {total_loss}")
