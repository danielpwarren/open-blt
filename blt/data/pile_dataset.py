import io
import json
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import zstandard as zstd

class PileDataset(IterableDataset):
    """
    Streaming dataset for The Pile that dynamically reads and decompresses data.
    Uses IterableDataset to enable streaming without loading everything into memory.
    """
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        bos_id: int = 257,
        eos_id: int = 258,
        shuffle_files: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.shuffle_files = shuffle_files
        
        # Get list of all zst files in the directory
        self.file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.jsonl.zst')
        ])
        
        if seed is not None:
            random.seed(seed)
            
        self.buffer = []  # Buffer to store tokens that didn't fit in the last sequence
        
    def _get_tokens_from_text(self, text: str) -> List[int]:
        """Convert text to token sequence with BOS and EOS tokens."""
        return [self.bos_id] + list(text.encode('utf-8')) + [self.eos_id]
    
    def _process_tokens(self, tokens: List[int]) -> List[torch.Tensor]:
        """Process a list of tokens into seq_len chunks."""
        sequences = []
        
        # Add any leftover tokens from the previous text
        if self.buffer:
            tokens = self.buffer + tokens
            self.buffer = []
            
        # Create full sequences of seq_len tokens
        for i in range(0, len(tokens) - 1, self.seq_len):
            chunk = tokens[i:i + self.seq_len]
            if len(chunk) == self.seq_len:
                sequences.append(torch.tensor(chunk, dtype=torch.long))
            else:
                self.buffer = chunk  # Save incomplete chunk for next text
                
        return sequences

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.file_paths
        
        if worker_info is not None:
            # Split files among workers
            per_worker = int(np.ceil(len(files_to_process) / worker_info.num_workers))
            worker_id = worker_info.id
            files_to_process = files_to_process[
                worker_id * per_worker : (worker_id + 1) * per_worker
            ]
            
        if self.shuffle_files:
            files_to_process = files_to_process.copy()
            random.shuffle(files_to_process)
            
        for file_path in files_to_process:
            with open(file_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    for line in text_stream:
                        try:
                            data = json.loads(line)
                            text = data.get('text', '')
                            if not text:
                                continue
                                
                            tokens = self._get_tokens_from_text(text)
                            sequences = self._process_tokens(tokens)
                            
                            for seq in sequences:
                                # Create input and target sequences
                                input_seq = seq[:-1]
                                target_seq = seq[1:]
                                yield input_seq, target_seq
                                
                        except Exception as e:
                            print(f"Error processing line in {file_path}: {e}")
                            continue

class PileValidationDataset(Dataset):
    """
    Dataset for validation data from The Pile.
    Loads a fixed set of validation examples into memory.
    """
    def __init__(
        self,
        val_file: str,
        seq_len: int,
        bos_id: int = 257,
        eos_id: int = 258,
        max_samples: int = 1000,  # Limit number of validation samples
    ):
        super().__init__()
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        self.sequences = []
        
        # Load validation samples
        with open(val_file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for i, line in enumerate(text_stream):
                    if i >= max_samples:
                        break
                        
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        if not text:
                            continue
                            
                        # Convert text to tokens
                        tokens = [self.bos_id] + list(text.encode('utf-8')) + [self.eos_id]
                        
                        # Take first seq_len tokens if text is too long
                        if len(tokens) > self.seq_len:
                            tokens = tokens[:self.seq_len]
                        
                        # Pad if necessary
                        if len(tokens) < self.seq_len:
                            tokens = tokens + [self.eos_id] * (self.seq_len - len(tokens))
                            
                        self.sequences.append(torch.tensor(tokens, dtype=torch.long))
                        
                    except Exception as e:
                        print(f"Error processing validation line {i}: {e}")
                        continue
                        
        print(f"Loaded {len(self.sequences)} validation sequences")
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]  # Return input and target sequences 