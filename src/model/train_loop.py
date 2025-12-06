import math
import os
import time

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torch.nn.functional as F
from gpt2 import GPT, GPTConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class DataLoaderNPY:
    def __init__(self, npy_file, B, T, process_rank, num_processes, train_split=0.9):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        self.tokens = np.load(npy_file).astype(np.int32)

        split_idx = int(len(self.tokens) * train_split)
        self.train_tokens = self.tokens[:split_idx]
        self.val_tokens = self.tokens[split_idx:]

        self.current_tokens = self.train_tokens
        self.current_position = self.B * self.T * self.process_rank

    def set_split(self, split):
        self.current_tokens = self.train_tokens if split == "train" else self.val_tokens
        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        if len(self.current_tokens) < B * T + 1:
            effective_B = max(1, len(self.current_tokens) // (T + 1))
            buf = self.current_tokens[: effective_B * T + 1]
            x = torch.from_numpy(buf[:-1]).view(effective_B, T).long()
            y = torch.from_numpy(buf[1:]).view(effective_B, T).long()
            return x, y

        buf = self.current_tokens[
            self.current_position : self.current_position + B * T + 1
        ]

        if len(buf) < B * T + 1:
            needed = (B * T + 1) - len(buf)
            buf = np.concatenate([buf, self.current_tokens[:needed]])
            self.current_position = needed
        else:
            self.current_position += B * T * self.num_processes

        if self.current_position + (B * T + 1) > len(self.current_tokens):
            self.current_position = self.B * self.T * self.process_rank

        x = torch.from_numpy(buf[:-1]).view(B, T).long()
        y = torch.from_numpy(buf[1:]).view(B, T).long()
        return x, y


enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288
B = 64
T = 256
assert total_batch_size % (B * T * ddp_world_size) == 0, (
    "make sure total_batch_size is divisible by B * T * ddp_world_size"
)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

data_loader = DataLoaderNPY(
    npy_file="./src/model/greentext_data/greentext.npy",
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
)

torch.set_float32_matmul_precision("high")

model = GPT(
    GPTConfig(block_size=256, vocab_size=50304, n_layer=4, n_head=4, n_embd=256)
)

model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device_type
)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

# Training loop
for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    if step % 250 == 0 or last_step:
        model.eval()
        data_loader.set_split("val")
        data_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = data_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    model.train()
    data_loader.set_split("train")
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = data_loader.B * data_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

model.save(model.state_dict(), "./weights")

if ddp:
    destroy_process_group()
