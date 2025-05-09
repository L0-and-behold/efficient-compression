import lzma
import os
from src.Datasets.Wiki40BDataset import WikipediaDatasets
from src.Datasets.Wiki40BDataset import ByteWikipediaDataset

def get_lzma_compressed_length(data: bytes, chunk_size: int):
    """
    Compress data using the LZMA2 algorithm via the .xz format.

    LZMA2 is the default filter used in the .xz format, which is the default
    format for the lzma module.
    Reference: https://docs.python.org/3/library/lzma.html#compression-formats
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

def run_lzma_compression_benchmark(datasets, subset_size: int, chunk_size: int):
    """
    Load data from the dataset, put in a single bytes object and pass to `get_lzma_compressed_length`

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

def main():
    """
    Run LZMA2 benchmarking for 
    - the dataset sizes 16MB: 16384000, 50MB: 50003968, 300MB: 299991040, 9.3GB: 9307817984

    Ensure that the dataset is stored locally in file 'processed_wiki_dataset.pt'
        -> see readme.md or src/Datasets/Wiki40BDatasets.py on how to download the dataset
    """

    dataset_local_path = os.path.join(os.getcwd(), 'processed_wiki_dataset.pt')
    sequence_length = 2048
    datasets = WikipediaDatasets.load_dataset(dataset_local_path, sequence_length)

    for dataset_sizes in [16384000, 50003968, 299991040, 9307817984]:
        run_lzma_compression_benchmark(datasets, dataset_sizes, sequence_length)

    print("done")


if __name__ == "__main__":
    main()




#### Results
