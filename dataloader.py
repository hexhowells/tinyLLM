import glob
import torch
from torch.utils.data import IterableDataset, Dataset
import pyarrow.parquet as pq
import numpy as np


class FineWebDataset(IterableDataset):
    def __init__(
            self,
            data_dir: str,
            seq_len: int = 1024
        ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        
        self.files = sorted(glob.glob(f"{data_dir}/**/*.npy", recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")


    def _get_worker_shards(self) -> list:
        """Splits numpy shards across DDP ranks and DataLoader workers."""
        worker_info = torch.utils.data.get_worker_info()
        
        if torch.distributed.is_initialized():  # used for multi-GPU setup
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # shard splitting across GPUs (ranks)
        rank_files = self.files[rank::world_size]  

        # shard splitting across DataLoader workers per GPU
        if worker_info is None:
            return rank_files  # single process, load data as is
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            num_streams = world_size * num_workers
            assert len(self.files) > num_streams, \
                f"Not enough shards ({len(self.files)}) for the given streams ({num_streams}), reduce num_workers"

            return rank_files[worker_id::num_workers]  # multi-process, split rank_files across workers


    def __iter__(self):
        shards = self._get_worker_shards()
        chunk_size = self.seq_len + 1
        unused_tokens = None

        for shard_path in shards:
            tokens = np.load(shard_path)
            tokens = torch.from_numpy(tokens).to(torch.long)

            if unused_tokens is not None:
                tokens = torch.cat([unused_tokens, tokens])
                unused_tokens = None

            total_tokens = tokens.size(0)

            idx = 0
            while idx + chunk_size <= total_tokens:
                chunk = tokens[idx:idx+chunk_size]
                yield chunk[:-1], chunk[1:]

                idx += self.seq_len

            if idx < total_tokens:
                unused_tokens = tokens[idx:]


class SmolTalkDataset(Dataset):
    def __init__(
            self, 
            data_dir: str,
            tokenizer,
            seq_len: int = 1024
        ) -> None:
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.seq_len = seq_len
                
        files = sorted(glob.glob(f"{data_dir}/**/*.parquet", recursive=True))
        if not files:
            raise FileNotFoundError(f"No .parquet shards found in {data_dir}")

        self.conversations: list[dict] = []

        for file in files:
            table = pq.read_table(file, columns=['messages'])
            messages = table['messages'].to_pylist()
            self.conversations.extend(messages)


    def __len__(self):
        return len(self.conversations)


    def _apply_chat_template(self, messages):
        text = ""
        for message in messages:
            if message["role"] == "system": continue
            text += f'<|{message["role"]}|>\n{message["content"]}\n'

        text += '<|assistant|>\n'

        return text


    def __getitem__(self, idx):
        messages = self.conversations[idx]

        prompt = self._apply_chat_template(messages[:-1])

        answer = messages[-1]['content'] + self.tokenizer.eos_token

        prompt_tokens = self.tokenizer.encode(prompt)
        answer_tokens = self.tokenizer.encode(answer)

        full_tokens = prompt_tokens + answer_tokens

        if len(full_tokens) > self.seq_len+1:
            full_tokens = full_tokens[:self.seq_len+1]

        x = torch.tensor(full_tokens[:-1], dtype=torch.long)
        y = torch.tensor(full_tokens[1:], dtype=torch.long)

        prompt_length = min(len(prompt_tokens)-1, len(y))
        if prompt_length > 0:
            y[:prompt_length] = -1

        return x, y


def sft_collate_fn(batch, pad_token_id):
    xs, ys = zip(*batch)
    x_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_token_id)
    y_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)

    return x_padded, y_padded
