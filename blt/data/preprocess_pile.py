import argparse
import io
import json
import os

import numpy as np
import zstandard as zstd


def preprocess_pile(input_dir, output_file, bos_id=257, eos_id=258, max_files=None):
    """
    Preprocess a directory of .jsonl.zst files by decompressing and extracting the "text" field,
    converting each record's text into token IDs with a BOS token at the beginning and an EOS token at the end,
    and writing the concatenated token stream to an output binary file (using uint16).

    For each record:
      tokens = [bos_id] + list(text.encode('utf-8')) + [eos_id]

    Args:
        input_dir (str): Directory containing .jsonl.zst files (e.g. the "train" folder).
        output_file (str): Path to the output binary file (e.g. "pile.bin").
        bos_id (int): Token ID for beginning-of-sequence (default 257).
        eos_id (int): Token ID for end-of-sequence (default 258).
        max_files (int, optional): Maximum number of files to process (for testing).
    """
    file_list = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".jsonl.zst")
        ]
    )
    if max_files is not None:
        file_list = file_list[:max_files]

    total_records = 0
    with open(output_file, "wb") as out_f:
        for file_path in file_list:
            print(f"Processing {file_path}...")
            with open(file_path, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                stream = dctx.stream_reader(f)
                text_stream = io.TextIOWrapper(stream, encoding="utf-8")
                for line in text_stream:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if text:
                            # Convert text to token IDs: bytes remain as is.
                            token_list = (
                                [bos_id] + list(text.encode("utf-8")) + [eos_id]
                            )
                            arr = np.array(token_list, dtype=np.uint16)
                            arr.tofile(out_f)
                            total_records += 1
                    except Exception as e:
                        print(f"Skipping a line due to error: {e}")
            print(f"Finished processing {file_path}.")
    print(f"Preprocessing complete. Total records processed: {total_records}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a directory of .jsonl.zst files from The Pile dataset into a binary token file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .jsonl.zst files (e.g., the 'train' folder).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output binary file to write the token stream (e.g., 'pile.bin').",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional maximum number of files to process (for testing).",
    )
    args = parser.parse_args()
    preprocess_pile(args.input_dir, args.output_file, max_files=args.max_files)


if __name__ == "__main__":
    main()
