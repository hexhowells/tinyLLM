"""This script pre-tokenises a .paraquet dataset and stores it as .npy shards to speed up dataloading during training."""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Pre-tokenise text in a dataset into various numpy binary shards."
    )
    parser.add_argument(
        "-i",
        "--dataset_dir",  # /media/datasets/fineweb-edu/sample/10BT
        type=Path,
        help="Directory of the .paraquet files to load",
    )
    parser.add_argument(
        "-o",
        "--output_dir",  # /media/datasets/fineweb-edu/processed/
        type=Path,
        help="Base directory path to store the shards",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="The tokenizer name to use to tokenize the data (loaded from HuggingFace)",
    )
    parser.add_argument(
        "-s",
        "--shard_size",
        type=int,
        default=100 * 1024 * 1024,  # ~100M tokens (2MB shard files)
        help="The maximum number of tokens each shard file should contain",
    )
    return parser.parse_args()


def write_shard(tokens: list[int], shard_dir: str):
    """
    Write N tokens to a single shard file

    Args:
        tokens: list of tokens to write
        shard_dir: shard filepath to write file to
    
    Returns:
        number of tokens written to the shard file
    """
    arr = np.array(tokens, dtype=np.uint16)
    tokens_written = len(arr)

    np.save(shard_dir, arr)
    print(f"Wrote {tokens_written} tokens to shard file {shard_dir}")

    return tokens_written


def main(
        data_dir: str, 
        output_dir: str,
        tokenizer_name: str,
        shard_size: int
    ) -> None:
    """
    Entry function to pre-tokenise a dataset stored in paraquet shards

    Args:
        data_dir: the directory of the .paraquet files to process
        output_dir: the output folder to store the pre-processed shards
        tokenizer_name: the name of the HuggingFace tokenizer to use
        shard_size: the number of tokens each shard should have
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(f"{data_dir}/**/*.parquet", recursive=True))
    assert files is not None, f"No .parquet files found in location {data_dir}"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)

    token_buffer = []
    shard_index = 0

    for file in files:
        parquet_file = pq.ParquetFile(file)

        for rg_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(rg_idx, columns=["text"])
            texts = table["text"].to_pylist()

            for text in texts:
                if not text.strip():
                    continue

                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokens.append(tokenizer.eos_token_id)
                token_buffer.extend(tokens)

                while len(token_buffer) >= shard_size:
                    shard_tokens = token_buffer[:shard_size]
                    shard_dir = os.path.join(output_dir, f"shard_{shard_index:05d}.npy")
                    tokens_written = write_shard(shard_tokens, shard_dir)
                    token_buffer = token_buffer[tokens_written:]
                    shard_index += 1

    if len(token_buffer) > 0:
        shard_dir = os.path.join(output_dir, f"shard_{shard_index:05d}.npy")
        write_shard(token_buffer, shard_dir)


if __name__ == "__main__":
    args = parse_args()

    main(
        data_dir=args.dataset_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        shard_size=args.shard_size
    )
