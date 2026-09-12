"""
Vibe-coded benchmark tool to compare the new dataloader to the original one developed in tinyGPT

Requires the old dataloader class to be copied into dataloader.py and renamed
"""
import time
import torch
from typing import Iterable
from tinyLLM.dataloader import FineWebDataset, FineWebDatasetOld
from transformers import AutoTokenizer


def benchmark_dataloader(
    dataloader: Iterable, 
    name: str = "DataLoader", 
    num_batches: int = 500, 
    warmup_batches: int = 20,
    seq_len: int = 1024
):
    """
    Benchmarks throughput and latency of a dataloader.
    """
    print(f"--- Benchmarking: {name} ---")
    
    # Initialize iterator
    t0 = time.perf_counter()
    iterator = iter(dataloader)
    init_time = time.perf_counter() - t0
    print(f"Initialization / Worker Startup: {init_time:.4f}s")
    
    # Warmup phase (exclude from throughput calculations)
    for _ in range(warmup_batches):
        try:
            next(iterator)
        except StopIteration:
            print("Dataset exhausted during warmup.")
            return

    # Benchmark phase
    total_tokens = 0
    total_samples = 0
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for i in range(num_batches):
        try:
            batch = next(iterator)
            
            # Assuming standard (x, y) tuple where x is the input tensor
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            # Extract batch size (handles both batched and unbatched outputs)
            batch_size = x.size(0) if x.dim() > 1 else 1
            
            total_samples += batch_size
            total_tokens += (batch_size * seq_len)
            
        except StopIteration:
            num_batches = i
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    
    if num_batches == 0:
        print("No batches processed during benchmark phase.\n")
        return

    # Metrics
    batches_per_sec = num_batches / elapsed
    samples_per_sec = total_samples / elapsed
    tokens_per_sec = total_tokens / elapsed
    ms_per_batch = (elapsed / num_batches) * 1000

    print(f"Processed {num_batches} batches in {elapsed:.2f}s")
    print(f"Latency:        {ms_per_batch:.2f} ms/batch")
    print(f"Throughput:     {batches_per_sec:.2f} batches/s")
    print(f"                {samples_per_sec:.2f} samples/s")
    print(f"                {tokens_per_sec:,.0f} tokens/s\n")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.model_max_length = int(1e30) 
    old_dataset = FineWebDatasetOld(data_dir="/media/datasets/fineweb_10BT", tokenizer=tokenizer, seq_len=1024) 
    old_loader = DataLoader(old_dataset, batch_size=32, num_workers=4)

    new_dataset = FineWebDataset(data_dir="/media/datasets/fineweb-edu/processed/", seq_len=1024)
    new_loader = DataLoader(new_dataset, batch_size=32, num_workers=4)

    benchmark_dataloader(old_loader, name="Old Parquet Loader", num_batches=1000, seq_len=1024)
    benchmark_dataloader(new_loader, name="New NumPy Loader", num_batches=1000, seq_len=1024)