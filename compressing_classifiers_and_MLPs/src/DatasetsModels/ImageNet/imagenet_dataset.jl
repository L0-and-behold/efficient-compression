## modified version of: https://lux.csail.mit.edu/stable/tutorials/advanced/3_ImageNet
#
#
# ==============================================================================
# Constants and utilities
# ==============================================================================
#

"""
List of known corrupted ImageNet JPEG files that should be skipped when
building the dataset.

These filenames are excluded during dataset indexing to avoid runtime
decode errors.
"""
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

"""
Channel-wise mean used for ImageNet normalization.
"""
const IMAGENET_MEAN = (0.485f0, 0.456f0, 0.406f0)

"""
Channel-wise standard deviation used for ImageNet normalization.
"""
const IMAGENET_STD  = (0.229f0, 0.224f0, 0.225f0)

"""
Number of color channels expected for ImageNet images.
"""
const CHANNELS = 3

# ==============================================================================
# ImageNet dataset (JPEG-based, runtime decoding)
# ==============================================================================

"""
Data augmentation transform that ensures grayscale images are converted
to 3-channel RGB by replicating the single channel.
"""
struct MakeColoredImage <: DataAugmentation.Transform end

"""
Simple file-based dataset for ImageNet-style directory layouts.

# Fields
- `files`: Vector of absolute file paths to image files.
- `labels`: Vector of integer class labels in the range `0:999`.
- `augment`: DataAugmentation pipeline applied on-the-fly during indexing.
"""
struct FileDataset
    files
    labels
    augment
end

"""
    load_imagenet1k(base_path::String, split::Symbol)

Load ImageNet-1k image file paths and labels for a given split.

# Arguments
- `base_path`: Root directory containing `train/` and `val/` subdirectories.
- `split`: Either `:train` or `:val`.

# Returns
- `image_files::Vector{String}`: Absolute paths to all images in the split.
- `labels::Vector{Int}`: Corresponding class labels in the range `0:999`.

The directory structure is assumed to be:
```
base_path/
  train/
    n01440764/
    n01443537/
    ...
  val/
    n01440764/
    ...
```

Known corrupted files are automatically filtered out.
"""
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

"""
Return the default ImageNet image size (224) if no size is provided.
"""
default_image_size(_, ::Nothing) = 224

"""
Return the explicitly requested image size.
"""
default_image_size(_, size::Int) = size

"""
Apply method for `MakeColoredImage`.

Ensures that the output image has exactly three channels by converting
grayscale images to RGB via channel replication.
"""
function DataAugmentation.apply(
    ::MakeColoredImage, item::DataAugmentation.AbstractArrayItem; randstate=nothing
)
    data = itemdata(item)
    (ndims(data) == 2 || size(data, 3) == 1) && (data = cat(data, data, data; dims=Val(3)))
    return DataAugmentation.setdata(item, data)
end

"""
Return the number of samples in a `FileDataset`.
"""
Base.length(dataset::FileDataset) = length(dataset.files)

"""
    getindex(dataset::FileDataset, i::Int)

Load and augment the `i`-th image from disk.

# Returns
- `image::Array{Float32,3}`: Image tensor of shape `(H, W, C)`.
- `label::OneHotVector`: One-hot encoded label over classes `0:999`.

All images are loaded lazily and transformed at access time.
"""
function Base.getindex(dataset::FileDataset, i::Int)
    img = Image(FileIO.load(dataset.files[i]))
    aug_img = itemdata(DataAugmentation.apply(dataset.augment, img))

    # each image is a (H, W, C) tensor
    @assert ndims(aug_img) == 3 "Each image should be a 3-tensor but got length $(ndims(aug_img))"
    @assert size(aug_img, 3) == CHANNELS "Expected 3rd entry to be color-channels but got dimension $(size(aug_img, 1))"
    @assert size(aug_img, 1) > 8 && size(aug_img, 2) > 8 "Expected the 1st and 2nd dimensions to be height and width"

    return aug_img, OneHotArrays.onehot(dataset.labels[i], 0:999)
end

"""
    construct_online_dataloaders(
        base_path::String,
        train_batchsize,
        val_batchsize;
        crop_size::Int=224,
        dev=gpu_device(),
        shuffle=true,
        random_crop=true,
    )

Construct ImageNet training and validation dataloaders using on-the-fly
JPEG decoding and data augmentation.

# Arguments
- `base_path`: Root ImageNet directory.
- `train_batchsize`: Batch size for the training loader.
- `val_batchsize`: Batch size for the validation loader.
- `crop_size`: Final spatial resolution of images (e.g. 224).
- `dev`: Device mapping function (e.g. `gpu_device()` or `cpu_device()`).
- `shuffle`: Whether to shuffle the training dataset.
- `random_crop`: Whether to apply random resized cropping during training.

# Returns
- `train_loader`: Device-mapped training `DataLoader`.
- `val_loader`: Device-mapped validation `DataLoader`.

This function performs runtime image decoding and is intended for
large-scale ImageNet training without preprocessed binaries.
"""
function construct_online_dataloaders(
        base_path::String,
        train_batchsize,
        val_batchsize;
        crop_size::Int=224,
        dev=gpu_device(),
        shuffle=true,
        random_crop=true,
        )
    println("=> creating dataloaders.")

    @inline MaybeRandomResizeCrop(enabled::Bool, crop_size::Int) =
        enabled ? RandomResizeCrop((crop_size, crop_size)) : identity

    train_augment =
        ScaleFixed((256, 256)) |>
        MaybeRandomResizeCrop(random_crop, crop_size) |>
        PinOrigin() |>
        ImageToTensor() |>
        MakeColoredImage() |>
        ToEltype(Float32) |>
        Normalize(IMAGENET_MEAN, IMAGENET_STD)

    train_files, train_labels = load_imagenet1k(base_path, :train)
    train_dataset = FileDataset(train_files, train_labels, train_augment)

    val_augment =
        ScaleFixed((crop_size, crop_size)) |>
        PinOrigin() |>
        ImageToTensor() |>
        MakeColoredImage() |>
        ToEltype(Float32) |>
        Normalize(IMAGENET_MEAN, IMAGENET_STD)

    val_files, val_labels = load_imagenet1k(base_path, :val)
    val_dataset = FileDataset(val_files, val_labels, val_augment)

    train_dataloader = DataLoader(
        train_dataset;
        batchsize=train_batchsize,
        partial=false,
        collate=true,
        shuffle=shuffle,
        parallel=false,
    )
    val_dataloader = DataLoader(
        val_dataset;
        batchsize=val_batchsize,
        partial=true,
        collate=true,
        shuffle=false,
        parallel=false,
    )

    # Validate dataloader output format eagerly
    x, y = first(train_dataloader)
    assert_hwcn(x)
    @assert size(x, 4) == length(onecold(y)) "Mismatch between batch size and labels."

    return dev(train_dataloader), dev(val_dataloader)
end


"""
Returns a function with the same signature as CIFAR_data and MNIST_data,
configured to use a specific ImageNet dataloader constructor (online, chunked, or toy).

# Arguments
- `root::String`: Path to ImageNet data directory
- `dataloader_constructor::Function`: One of `construct_online_dataloaders`, 
  `construct_chunked_dataloaders`, or `construct_toy_dataloaders`
- `crop_size::Int=224`: Image crop size
- `dev`: Device function (gpu_device() or cpu_device())

# Returns
A function `(batch_size::Int) -> (train_set, val_set, test_set)` where `test_set` is always `nothing`
"""
function imagenet_data_function(
    root::String,
    dataloader_constructor::Function;
    crop_size::Int=224,
    dev=gpu_device()
    )::Function

    function imagenet_dataset(batch_size::Int)

        train_set, val_set = dataloader_constructor(root, batch_size, batch_size; crop_size=crop_size, dev=dev)
        test_set = nothing

        return train_set, val_set, test_set
    end

    return imagenet_dataset
end
