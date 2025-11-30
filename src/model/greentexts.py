import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  
from tqdm import tqdm  

local_dir = "greentext_data"
shard_size = int(1e7)  
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

print("Downloading DarwinAnim8or/greentext dataset from Hugging Face...")
ds = load_dataset(
    'json',
    data_files='hf://datasets/DarwinAnim8or/greentext/greentexts.jsonl',
    split='train'
)
print(f"Dataset downloaded successfully! Total examples: {len(ds)}")
print(f"Dataset features: {ds.features}")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] 

def tokenize(doc):
    tokens = [eot]  
    if "text" in doc:
        text = doc["text"]
    elif "prompt" in doc and "completion" in doc:
        text = f"{doc['prompt']}\n{doc['completion']}"
    else:
        text = ""
    
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16) 
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, ds, chunksize=16):
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"greentext_{split}_{shard_index:06d}")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
    
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"greentext_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

print(f"\nDone! Saved {shard_index + 1} shards to {DATA_CACHE_DIR}")