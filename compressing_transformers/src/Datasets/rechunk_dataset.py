"""
Rechunk a saved WikipediaDatasets file to a new sequence length.

Usage (run from project root):
    python -m src.Datasets.rechunk_dataset --input /path/to/processed_wiki_dataset_2048.pt --target 1024
    python -m src.Datasets.rechunk_dataset --input /path/to/dir --source 2048 --target 512

--input can be either a .pt file path or a directory. When a directory is given,
--source is used to locate processed_wiki_dataset_<source>.pt inside it.
Output is written as processed_wiki_dataset_<target>.pt in the same directory.
"""
import argparse
import os

import torch
from tqdm import tqdm

from .Wiki40BDataset import ByteWikipediaDataset, WikipediaDatasets


def rechunk(byte_list, new_seq_len, label):
    stream = b''.join(byte_list)
    n = len(stream)
    total_full = n // new_seq_len
    chunks = []
    for i in tqdm(range(total_full), desc=f"  {label}", unit="chunk"):
        chunks.append(stream[i * new_seq_len:(i + 1) * new_seq_len])
    remainder = stream[total_full * new_seq_len:]
    if remainder:
        chunks.append(remainder + b'\0' * (new_seq_len - len(remainder)))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to source .pt file, or directory (requires --source)")
    parser.add_argument("--source", type=int, default=None,
                        help="Source sequence length when --input is a directory")
    parser.add_argument("--target", required=True, type=int,
                        help="Target sequence length, e.g. 1024 or 512")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        if args.source is None:
            parser.error("--source is required when --input is a directory")
        source_path = os.path.join(args.input, f"processed_wiki_dataset_{args.source}.pt")
        out_dir = args.input
    else:
        source_path = args.input
        out_dir = os.path.dirname(os.path.abspath(args.input))

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    print(f"Loading {source_path} ...")
    source = torch.load(source_path, weights_only=False)
    print(f"Source sequence length: {source.sequence_length}")
    print(f"  train:      {len(source.train.data):>8,} chunks")
    print(f"  validation: {len(source.validation.data):>8,} chunks")
    print(f"  test:       {len(source.test.data):>8,} chunks\n")

    new_seq_len = args.target
    datasets = WikipediaDatasets(
        train=ByteWikipediaDataset(rechunk(source.train.data, new_seq_len, "train"), 'train'),
        validation=ByteWikipediaDataset(rechunk(source.validation.data, new_seq_len, "validation"), 'validation'),
        test=ByteWikipediaDataset(rechunk(source.test.data, new_seq_len, "test"), 'test'),
        sequence_length=new_seq_len,
    )

    out_path = os.path.join(out_dir, f"processed_wiki_dataset_{new_seq_len}.pt")
    print(f"\nSaving {out_path} ...")
    torch.save(datasets, out_path)
    print(f"  train:      {len(datasets.train.data):>8,} chunks")
    print(f"  validation: {len(datasets.validation.data):>8,} chunks")
    print(f"  test:       {len(datasets.test.data):>8,} chunks")
    print("\nDone.")


if __name__ == "__main__":
    main()
