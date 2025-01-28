import math
import os
from typing import List, Tuple
from process.Dataset import SequenceDataset
from tqdm import tqdm
from process.Tokenizer import BPETokenizer
from datasets import load_dataset
from model.helper import causal_mask
from model.ModelArgs import TransformerArgs
from model.Transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
import accelerate
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from multiprocessing import Pool, cpu_count
from torch.amp import GradScaler, autocast
from accelerate.utils import set_seed
from torch.distributed import destroy_process_group
import atexit


CONTEXT_WINDOW = 512
TARGET_WINDOW = 8
BATCH_SIZE = 32
EPOCHS = 10
BASE_LR = 5e-4
SEED = 42
WARMUP_STEPS = 1000
DATASET = "prithivMLmods/System-Response-100K"

BLUE_ANSI = "\033[94m"
RESET_ANSI = "\033[0m"
GRAY_ANSI = "\033[90m"


def cleanup():
    if torch.distributed.is_initialized():
        destroy_process_group()


atexit.register(cleanup)


def info(*args):
    print(BLUE_ANSI, *args, RESET_ANSI)


def debug(*args):
    print(GRAY_ANSI, *args, RESET_ANSI)


accelerator = accelerate.Accelerator()
if accelerator.is_main_process:
    info("Setting seed for RNG synchronization")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


device = accelerator.device
info("Using device:", device)

dataset = load_dataset(DATASET)["train"]
train = dataset["question"]  # question column also includes the answer
info("Loaded dataset with", len(train), "samples.")

tokenizer = BPETokenizer()
tokenizer.train(train)
if not os.path.exists("tokenizer.json"):
    tokenizer.save("tokenizer.json")
else:
    tokenizer.load("tokenizer.json")
info("Tokenizer saved to tokenizer.json.")
debug("Sample enconding", tokenizer.encode(train[0]))


def encode_sample(sample):
    return tokenizer.encode(sample)


info("Encoding dataset...")
num_workers = cpu_count()
with Pool(num_workers) as pool:
    data = list(tqdm(pool.imap(encode_sample, train), total=len(train)))

info("Data tokenized with tokenizer.")
dataset = SequenceDataset(CONTEXT_WINDOW, TARGET_WINDOW, data, verbose=True)
info("Dataset created with", len(dataset), "samples.")


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a batch of samples into padded tensors.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A batch of context-target pairs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batched and padded context and target tensors.
    """
    contexts, targets = zip(*batch)
    context_padded = torch.stack(contexts)
    target_padded = torch.stack(targets)
    return context_padded, target_padded


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

info("Dataloader created with", len(dataloader), "batches.")

for batch in dataloader:
    context, target = batch

    ex_context: torch.Tensor = context[0]
    ex_target: torch.Tensor = target[0]

    debug("Example context:", tokenizer.decode(ex_context.tolist()))
    debug("Example target:", tokenizer.decode(ex_target.tolist()))
    break


model_args = TransformerArgs(
    vocab_size=tokenizer.inner_tokenizer().get_vocab_size(),
    d_model=512,
    d_ff=2048,
    n_head=8,
    num_latents=64,
    num_layers=6,
    dropout=0.1,
    max_len=CONTEXT_WINDOW,
)

model = Transformer(model_args).to(device)
human_readable = f"{sum(p.numel() for p in model.parameters()):,}"
info("Model created with", human_readable, "parameters.")

optimizer = optim.AdamW(model.parameters(), lr=BASE_LR)
scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, len(dataloader))

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
torch.autograd.set_detect_anomaly(True)
debug(f"Autocast device: {str(device)}")
scaler = GradScaler(device=str(device), enabled=True)

model, optimizer, dataloader, scaler = accelerator.prepare(
    model, optimizer, dataloader, scaler
)
info("Starting training...")


def train_step(context, target):
    optimizer.zero_grad()

    mask = causal_mask(seq_len=context.size(1), num_latents=model_args.num_latents).to(
        device
    )

    with autocast(device_type=str(device)):
        output = model(context, mask)
        output = output[:, -target.size(1) :, :]
        loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
        if torch.isnan(loss).any():
            debug("Output:", output)
            debug("Target:", target)
            debug("Loss:", loss)
            debug("Context:", context)
            debug("Mask:", mask)
            raise ValueError("Loss is NaN")

    accelerator.backward(loss)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    return loss.item()


last_loss = float("inf")
for epoch in range(EPOCHS):
    info(f"Epoch {epoch + 1} of {EPOCHS}")
    model.train()

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for i, batch in progress_bar:
        context, target = batch
        loss = train_step(context, target)
        if math.isnan(loss):
            break
        if loss < last_loss:
            last_loss = loss
            torch.save(model.state_dict(), "model.pth")
        last_loss = loss
        progress_bar.set_postfix({"Loss": f"{loss:.4f}"})

    info(f"Epoch {epoch + 1} completed with loss {loss:.4f}.")
info("Training completed.")
