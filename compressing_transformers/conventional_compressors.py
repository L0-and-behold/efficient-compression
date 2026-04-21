"""
Benchmark LZMA2 and Zstandard compression on the Wikipedia dataset.

Usage:
    python conventional_compressors.py
    python conventional_compressors.py --dataset /path/to/processed_wiki_dataset_2048.pt
    python conventional_compressors.py --dataset /path/to/dir --seq-len 512
    python conventional_compressors.py --dataset /path/to/dir --sizes 299991040 1232000000
    python conventional_compressors.py --workers 8

Results are written to output/conventional_compressors_benchmark.out.
"""
import argparse
import lzma
import multiprocessing as mp
import os
import sys

import zstandard as zstd
from tqdm import tqdm

from src.Datasets.Wiki40BDataset import WikipediaDatasets

DEFAULT_SIZES = [6159990784, 299991040, 1232000000]
ZSTD_LEVEL = 19

# Populated in the main process before forking; workers inherit via copy-on-write.
_shared_data: bytes = b""


def _compress_segment_lzma(args):
    start, end, chunk_size = args
    data = _shared_data[start:end]
    filters = [{"id": lzma.FILTER_LZMA2}]
    total = 0
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        lzc = lzma.LZMACompressor(format=lzma.FORMAT_XZ, filters=filters)
        total += len(lzc.compress(chunk) + lzc.flush())
    return total


def _compress_segment_zstd(args):
    start, end, chunk_size, level = args
    data = _shared_data[start:end]
    compressor = zstd.ZstdCompressor(level=level)
    total = 0
    for i in range(0, len(data), chunk_size):
        total += len(compressor.compress(data[i:i + chunk_size]))
    return total


def _compress_parallel(data: bytes, chunk_size: int, worker_fn, extra_args: tuple,
                       n_workers: int, desc: str) -> int:
    global _shared_data
    _shared_data = data

    n = len(data)
    total_chunks = (n + chunk_size - 1) // chunk_size
    n_segments = max(n_workers * 20, 100)
    chunks_per_seg = max(1, total_chunks // n_segments)

    segments = []
    chunk_idx = 0
    while chunk_idx < total_chunks:
        start = chunk_idx * chunk_size
        end = min((chunk_idx + chunks_per_seg) * chunk_size, n)
        segments.append((start, end) + extra_args)
        chunk_idx += chunks_per_seg

    fn = _compress_segment_lzma if worker_fn == "lzma" else _compress_segment_zstd
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(fn, segments),
                            total=len(segments), desc=desc, unit="seg"))
    return sum(results)


def _load_subset(datasets, subset_size: int) -> bytes:
    subset = WikipediaDatasets.get_subset(0, subset_size, datasets.train)
    data = b''.join(subset.data)
    assert type(data) is bytes
    assert sum(len(c) for c in subset.data) == len(data)
    return data


def run_benchmarks(datasets, sizes, seq_len, n_workers, out):
    def log(line=""):
        print(line)
        print(line, file=out)

    for size in sizes:
        gb = size / 1e9
        log(f"\n=== Dataset size: {size:,} bytes ({gb:.3f} GB) ===")
        data = _load_subset(datasets, size)
        actual = len(data)
        log(f"Loaded: {actual:,} bytes ({actual / 1e9:.3f} GB)")

        compressed = _compress_parallel(data, seq_len, "lzma", (seq_len,),
                                        n_workers, "LZMA2")
        ratio = actual / compressed if compressed else 0
        bpb = compressed * 8 / actual if actual else 0
        log(f"  LZMA2      compressed: {compressed:>15,} bytes  ratio: {ratio:.4f}  bits/byte: {bpb:.4f}")

        compressed = _compress_parallel(data, seq_len, "zstd", (seq_len, ZSTD_LEVEL),
                                        n_workers, f"Zstd-{ZSTD_LEVEL}")
        ratio = actual / compressed if compressed else 0
        bpb = compressed * 8 / actual if actual else 0
        log(f"  Zstd-{ZSTD_LEVEL:>2}   compressed: {compressed:>15,} bytes  ratio: {ratio:.4f}  bits/byte: {bpb:.4f}")

    log("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=".",
                        help="Path to .pt file or directory (default: current directory)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length / chunk size (default: 512)")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help="Dataset subset sizes in bytes")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help=f"Parallel workers (default: {os.cpu_count()} = all CPUs)")
    parser.add_argument("--output", default="output/conventional_compressors_benchmark.out",
                        help="Output file path (default: output/conventional_compressors_benchmark.out)")
    args = parser.parse_args()

    if os.path.isdir(args.dataset):
        dataset_path = os.path.join(args.dataset, f"processed_wiki_dataset_{args.seq_len}.pt")
    else:
        dataset_path = args.dataset

    if not os.path.exists(dataset_path):
        sys.exit(f"Dataset not found: {dataset_path}")

    datasets = WikipediaDatasets.load_dataset(dataset_path, args.seq_len)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as out:
        print(f"seq_len={args.seq_len}  zstd_level={ZSTD_LEVEL}  workers={args.workers}  sizes={args.sizes}", file=out)
        run_benchmarks(datasets, args.sizes, args.seq_len, args.workers, out)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
