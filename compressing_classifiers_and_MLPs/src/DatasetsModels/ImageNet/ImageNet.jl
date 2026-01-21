module ImageNet
    using Lux, Random, Distributed, OneHotArrays, CUDA, DataAugmentation, FileIO, ImageIO, Images, JLD2, ProgressMeter, Printf

    using OneHotArrays
    using Lux: DistributedUtils, gpu_device, cpu_device
    using Base.Iterators: partition
    using MLUtils: DataLoader
    import Lux.DistributedUtils.DistributedDataContainer

    include("imagenet_dataset.jl")
    export imagenet_online_data, imagenet_chunked_data, preprocess_split, ChunkedImageNet, ChunkedBatch, DeviceDataLoader

    include("toy_imagenet_dataset.jl")
    export toy_imagenet_data

end #module