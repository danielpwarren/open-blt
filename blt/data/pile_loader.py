import os
import json
import zstandard as zstd
import torch
from torch.utils.data import IterableDataset

# This class is not yet used, it will be expanded on once
# training is tested and preprocessing is done.
class PileDataset(IterableDataset):
    def __init__(self, data_dir, seq_length=2048):
        self.seq_length = seq_length
        self.file_paths = self._get_file_paths(data_dir)
        self.current_bytes = []
        
    def _get_file_paths(self, data_dir):
        train_dir = os.path.join(data_dir, "train")
        return sorted([
            os.path.join(train_dir, f) for f in os.listdir(train_dir)
            if f.endswith(".jsonl.zst")
        ])
    
    def _load_next_file(self):
        if not self.file_paths:
            return
            
        path = self.file_paths.pop(0)
        dctx = zstd.ZstdDecompressor()
        with open(path, 'rb') as f:
            reader = dctx.stream_reader(f)
            while True:
                chunk = reader.read(16384)  # 16KB chunks
                if not chunk:
                    break
                self.current_bytes.extend(chunk)
                
                # Process complete JSON lines
                while b'\n' in self.current_bytes:
                    pos = self.current_bytes.index(b'\n')
                    line = bytes(self.current_bytes[:pos])
                    self.current_bytes = self.current_bytes[pos+1:]
                    
                    try:
                        text = json.loads(line.decode('utf-8'))['text']
                        yield from [ord(c) for c in text]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def __iter__(self):
        buffer = []
        byte_gen = self._load_next_file()
        
        while True:
            try:
                buffer.append(next(byte_gen))
                if len(buffer) >= self.seq_length:
                    seq = torch.tensor(buffer[:self.seq_length], dtype=torch.long)
                    buffer = buffer[self.seq_length:]
                    yield seq
            except StopIteration:
                if len(buffer) > 0:
                    # Pad final sequence if needed
                    pad_len = self.seq_length - len(buffer)
                    seq = torch.tensor(buffer + [0]*pad_len, dtype=torch.long)
                    yield seq
                break