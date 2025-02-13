import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryDataset(Dataset):
    """
    A dataset that memory-maps a binary file containing token IDs (stored as uint16)
    and splits it into input and target sequences of length seq_len.
    """

    def __init__(self, file_path, seq_len):
        self.seq_len = seq_len
        # Use memmap to avoid loading the entire file into memory.
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        # Each sample requires (seq_len + 1) tokens (for input and target).
        self.num_samples = (len(self.data) - 1) // self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        # Get a slice from the memory-mapped array.
        seq = self.data[start:end]
        # If we don't have enough tokens, pad with zeros.
        if len(seq) < self.seq_len + 1:
            seq = np.pad(seq, (0, self.seq_len + 1 - len(seq)), mode="constant")
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq
