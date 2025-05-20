import lzma
import os
from src.Datasets.Wiki40BDataset import WikipediaDatasets
from src.Datasets.Wiki40BDataset import ByteWikipediaDataset

def get_lzma_compressed_length(data: bytes, chunk_size: int) -> int:
    """Compress data using the LZMA2 algorithm via the .xz format and return compressed size.
    
    LZMA2 is the default filter used in the .xz format, which is the default
    format for the lzma module.
    
    Args:
        data: The bytes object to compress
        chunk_size: Size of chunks to process at a time
        
    Returns:
        The size of the compressed data in bytes
        
    Reference: 
        https://docs.python.org/3/library/lzma.html#compression-formats
    """
    
    # Explicitly define filter and format for LZMA2
    filters = [{"id": lzma.FILTER_LZMA2}]
    format = lzma.FORMAT_XZ

    print(f"Start compression with chunk size {chunk_size}")
    compressed_size = 0

    # Process in chunks of the specified size
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        # Create a new compressor for each chunk
        lzc = lzma.LZMACompressor(format=format, filters=filters)
        compressed_chunk = lzc.compress(chunk) + lzc.flush()
        compressed_size += len(compressed_chunk)

    return compressed_size

def run_lzma_compression_benchmark(datasets, subset_size: int, chunk_size: int) -> None:
    """Load data from the dataset, compile into a single bytes object and compress with LZMA.
    
    Args:
        datasets: WikipediaDatasets object containing the training data
        subset_size: Number of bytes to include in the subset for compression
        chunk_size: Size of chunks to use in the compression process
    """

    # load data as one big bytes object
    subset = WikipediaDatasets.get_subset(0, subset_size, datasets.train)
    
    # Build one big bytes object from the dataset subset
    data = b''.join(subset.data)

    # Assert object-type and number of bytes
    assert(type(data) is bytes)
    subset_length = 0
    for chunk in subset.data:
        subset_length += len(chunk)
    assert(subset_length == len(data))

    print(f"Originial size: {len(data)} bytes")
    # Compress using LZMA
    compressed_size = get_lzma_compressed_length(data, chunk_size)
    print(f"Compressed size: {compressed_size} bytes")

def main() -> None:
    """Run LZMA2 benchmarking for various dataset sizes.
    
    The benchmarks are run for the following dataset sizes:
    - 16MB: 16384000 bytes
    - 50MB: 50003968 bytes
    - 300MB: 299991040 bytes
    - 9.3GB: 9307817984 bytes
    
    Note:
        Ensure that the dataset is stored locally in file 'processed_wiki_dataset.pt'.
        See readme.md or src/Datasets/Wiki40BDatasets.py for instructions on downloading the dataset.
    """

    dataset_local_path = os.path.join(os.getcwd(), 'processed_wiki_dataset.pt')
    sequence_length = 2048
    datasets = WikipediaDatasets.load_dataset(dataset_local_path, sequence_length)

    for dataset_sizes in [16384000, 50003968, 299991040, 9307817984]:
        run_lzma_compression_benchmark(datasets, dataset_sizes, sequence_length)

    print("done")


if __name__ == "__main__":
    main()