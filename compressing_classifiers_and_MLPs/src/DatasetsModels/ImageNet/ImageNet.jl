"""
ImageNet dataset utilities for Lux-based training pipelines.

This module provides:
- Online ImageNet loading with runtime JPEG decoding and augmentation
- Offline (chunked) ImageNet preprocessing and loading

### Provided APIs
- `imagenet_online_data`: Runtime JPEG-based dataloaders
- `preprocess_split`: Offline preprocessing into chunked `.jld2` files
- `imagenet_chunked_data`: Dataloaders backed by preprocessed chunks
- `ChunkedImageNet`, `ChunkedBatch`, `DeviceDataLoader`: Internal dataset and loader types
- `toy_imagenet_data`: Lightweight ImageNet-like dataset for testing

### Notes
- Image tensors follow the HWCN layout `(H, W, C, N)`
- Random cropping for chunked datasets is applied at runtime
- Device transfer is handled via `Lux.gpu_device` / `cpu_device`

This module is adapted from:
https://lux.csail.mit.edu/stable/tutorials/advanced/3_ImageNet
"""
module ImageNet
    using Lux
    using Random, Distributed, Printf
    using OneHotArrays
    using CUDA
    using NNlib
    using DataAugmentation
    using FileIO, ImageIO, Images
    using JLD2
    using ProgressMeter
    using Statistics: mean

    using Lux: DistributedUtils, gpu_device, cpu_device
    using Base.Iterators: partition
    using MLUtils: DataLoader

    import Lux.DistributedUtils.DistributedDataContainer

    using ...Config: load_imagenet_config

    # Online (JPEG-based) ImageNet dataset
    include("imagenet_dataset.jl")

    # Offline preprocessing + chunked ImageNet dataset
    include("imagenet_preprocessed_dataset.jl")
    
    # Toy ImageNet (debug)
    include("toy_imagenet_dataset.jl")

    export imagenet_data_function, construct_online_dataloaders, construct_chunked_dataloaders, construct_toy_dataloaders, preprocess_split, ChunkedImageNet, ChunkedBatch, DeviceDataLoader, imagenet_preprocessed_data

end # module ImageNet