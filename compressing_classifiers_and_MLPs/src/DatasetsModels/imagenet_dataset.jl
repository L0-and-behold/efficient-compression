## modified version of: https://lux.csail.mit.edu/stable/tutorials/advanced/3_ImageNet

# We need the data to be in a specific format. See the
# [README.md](<unknown>/examples/ImageNet/README.md) for more details.

using DataAugmentation
import MLUtils: DataLoader, DeviceIterator
using Lux
using FileIO
using OneHotArrays

const IMAGENET_CORRUPTED_FILES = [
    "n01739381_1309.JPEG",
    "n02077923_14822.JPEG",
    "n02447366_23489.JPEG",
    "n02492035_15739.JPEG",
    "n02747177_10752.JPEG",
    "n03018349_4028.JPEG",
    "n03062245_4620.JPEG",
    "n03347037_9675.JPEG",
    "n03467068_12171.JPEG",
    "n03529860_11437.JPEG",
    "n03544143_17228.JPEG",
    "n03633091_5218.JPEG",
    "n03710637_5125.JPEG",
    "n03961711_5286.JPEG",
    "n04033995_2932.JPEG",
    "n04258138_17003.JPEG",
    "n04264628_27969.JPEG",
    "n04336792_7448.JPEG",
    "n04371774_5854.JPEG",
    "n04596742_4225.JPEG",
    "n07583066_647.JPEG",
    "n13037406_4650.JPEG",
    "n02105855_2933.JPEG",
    "ILSVRC2012_val_00019877.JPEG",
]

function load_imagenet1k(base_path::String, split::Symbol)
    @assert split in (:train, :val)
    full_path = joinpath(base_path, string(split))
    synsets = sort(readdir(full_path))
    @assert length(synsets) == 1000 "There should be 1000 subdirectories in $(full_path)."

    image_files = String[]
    labels = Int[]
    for (i, synset) in enumerate(synsets)
        filenames = readdir(joinpath(full_path, synset))
        filter!(x -> x ∉ IMAGENET_CORRUPTED_FILES, filenames)
        paths = joinpath.((full_path,), (synset,), filenames)
        append!(image_files, paths)
        append!(labels, repeat([i - 1], length(paths)))
    end

    return image_files, labels
end

# default_image_size(::Type{Vision.VisionTransformer}, ::Nothing) = 256
# default_image_size(::Type{Vision.VisionTransformer}, size::Int) = size
default_image_size(_, ::Nothing) = 224
default_image_size(_, size::Int) = size

struct MakeColoredImage <: DataAugmentation.Transform end

function DataAugmentation.apply(
    ::MakeColoredImage, item::DataAugmentation.AbstractArrayItem; randstate=nothing
)
    data = itemdata(item)
    (ndims(data) == 2 || size(data, 3) == 1) && (data = cat(data, data, data; dims=Val(3)))
    return DataAugmentation.setdata(item, data)
end

struct FileDataset
    files
    labels
    augment
end

Base.length(dataset::FileDataset) = length(dataset.files)

function Base.getindex(dataset::FileDataset, i::Int)
    img = Image(FileIO.load(dataset.files[i]))
    aug_img = itemdata(DataAugmentation.apply(dataset.augment, img))
    return aug_img, OneHotArrays.onehot(dataset.labels[i], 0:999)
end

function construct_dataloaders(base_path::String, train_batchsize, val_batchsize, image_size::Int; dev=gpu_device())
    println("=> creating dataloaders.")

    train_augment =
        ScaleFixed((256, 256)) |>
        # Maybe(DataAugmentation.FlipX, 0.5) |>
        RandomResizeCrop((image_size, image_size)) |>
        PinOrigin() |>
        ImageToTensor() |>
        MakeColoredImage() |>
        ToEltype(Float32) |>
        Normalize((0.485f0, 0.456f0, 0.406f0), (0.229f0, 0.224f0, 0.225f0))
    train_files, train_labels = load_imagenet1k(base_path, :train)

    train_dataset = FileDataset(train_files, train_labels, train_augment)

    val_augment =
        ScaleFixed((image_size, image_size)) |>
        PinOrigin() |>
        ImageToTensor() |>
        MakeColoredImage() |>
        ToEltype(Float32) |>
        Normalize((0.485f0, 0.456f0, 0.406f0), (0.229f0, 0.224f0, 0.225f0))
    val_files, val_labels = load_imagenet1k(base_path, :val)

    val_dataset = FileDataset(val_files, val_labels, val_augment)

    # if is_distributed
    #     train_dataset = DistributedUtils.DistributedDataContainer(
    #         distributed_backend, train_dataset
    #     )
    #     val_dataset = DistributedUtils.DistributedDataContainer(
    #         distributed_backend, val_dataset
    #     )
    # end

    train_dataloader = DataLoader(
        train_dataset;
        batchsize=train_batchsize, # ÷ total_workers,
        partial=false,
        collate=true,
        shuffle=true,
        parallel=true,
    )
    val_dataloader = DataLoader(
        val_dataset;
        batchsize=val_batchsize, # ÷ total_workers,
        partial=true,
        collate=true,
        shuffle=false,
        parallel=true,
    )

    return dev(train_dataloader), dev(val_dataloader)
end

function imagenet_data(base_path::String, train_batchsize, val_batchsize, image_size::Int; dev=gpu_device())
    train_set, val_set = construct_dataloaders(base_path, train_batchsize, val_batchsize, image_size; dev=dev)
    test_set = deepcopy(val_set)
    return train_set, val_set, test_set
end

# include("imagenet/imagenet_path.jl")
# train_set, val_set, test_set = imagenet_data(imagenet_path, 256, 256, 224; dev=gpu_device())

# typeof(train_set) <: DeviceIterator

