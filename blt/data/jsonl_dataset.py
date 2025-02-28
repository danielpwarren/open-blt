import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from blt.tokenizer.abstract_tokenizer import Tokenizer


class JsonlDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        tokenizer: Tokenizer,
        shuffle_files: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.shuffle_files = shuffle_files
        self.file_paths = sorted(
            [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.endswith(".jsonl")
            ]
        )
        if seed is not None:
            random.seed(seed)
        self.buffer = []

    def _get_tokens_from_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_bos=True, add_eos=True)

    def _process_tokens(self, tokens: List[int]) -> List[torch.Tensor]:
        sequences = []
        if self.buffer:
            tokens = self.buffer + tokens
            self.buffer = []

        for i in range(0, len(tokens) - 1, self.seq_len):
            chunk = tokens[i : i + self.seq_len]
            if len(chunk) == self.seq_len:
                sequences.append(torch.tensor(chunk, dtype=torch.long))
            else:
                self.buffer = chunk
        return sequences

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.file_paths

        if worker_info is not None:
            per_worker = int(np.ceil(len(files_to_process) / worker_info.num_workers))
            worker_id = worker_info.id
            files_to_process = files_to_process[
                worker_id * per_worker : (worker_id + 1) * per_worker
            ]

        if self.shuffle_files:
            files_to_process = files_to_process.copy()
            random.shuffle(files_to_process)

        for file_path in files_to_process:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if not text:
                            continue

                        tokens = self._get_tokens_from_text(text)
                        sequences = self._process_tokens(tokens)

                        for seq in sequences:
                            yield seq[:-1], seq[1:]
                    except Exception as e:
                        print(f"Error processing line in {file_path}: {e}")
                        continue


class JsonlValidationDataset(Dataset):
    def __init__(
        self,
        val_file: str,
        seq_len: int,
        tokenizer: Tokenizer,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.sequences = []

        val_file = os.path.expanduser(val_file)
        with open(val_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue

                    tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                    tokens = tokens[: self.seq_len] + [self.tokenizer.eos_id] * (
                        self.seq_len - len(tokens)
                    )

                    self.sequences.append(torch.tensor(tokens, dtype=torch.long))
                except Exception as e:
                    print(f"Error processing validation line {i}: {e}")
                    continue

        print(f"Loaded {len(self.sequences)} validation sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]


def calculate_total_steps(self, batch_size: int) -> int:
        def count_sequences_in_file(file_path: str) -> int:
            count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get('text', '')
                            if not text:
                                continue
                            # Calculate number of sequences this text will generate
                            tokens = self._get_tokens_from_text(text)
                            n_sequences = len(tokens) // self.seq_len
                            count += n_sequences
                        except Exception as e:
                            print(f"Error counting sequences in {file_path}: {e}")
                            continue
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
            return count

        total_sequences = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(count_sequences_in_file, self.file_paths)
            total_sequences = sum(results)

        return total_sequences // batch_size
