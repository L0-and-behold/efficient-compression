## modified version of: https://lux.csail.mit.edu/stable/tutorials/advanced/3_ImageNet
#
# NOTE:
# This file currently contains both the legacy JPEG-based ImageNet
# dataloader and a new chunked (preprocessed) ImageNet pipeline.
# The chunked pipeline is intended to replace the legacy loader
# once fully validated.

# ==============================================================================
# Constants and utilities
# ==============================================================================

# TODO: update README to reflect this, also that we should run julia with multiple threads.
# TODO: update slrum script to use several julia threads.

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

const IMAGENET_MEAN = (0.485f0, 0.456f0, 0.406f0)
const IMAGENET_STD  = (0.229f0, 0.224f0, 0.225f0)
const CHANNELS = 3

# ==============================================================================
# Legacy ImageNet dataset (JPEG-based, runtime decoding)
# ==============================================================================

struct MakeColoredImage <: DataAugmentation.Transform end

struct FileDataset
    files
    labels
    augment
end

# TODO(deprecation):
# This function is part of the legacy ImageNet pipeline that loads and
# decodes JPEGs at runtime. Once the chunked ImageNet pipeline is fully
# validated, this loader should be removed or replaced.
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

function DataAugmentation.apply(
    ::MakeColoredImage, item::DataAugmentation.AbstractArrayItem; randstate=nothing
)
    data = itemdata(item)
    (ndims(data) == 2 || size(data, 3) == 1) && (data = cat(data, data, data; dims=Val(3)))
    return DataAugmentation.setdata(item, data)
end

Base.length(dataset::FileDataset) = length(dataset.files)

function Base.getindex(dataset::FileDataset, i::Int)
    img = Image(FileIO.load(dataset.files[i]))
    aug_img = itemdata(DataAugmentation.apply(dataset.augment, img))

    # each image is a (H, W, C) tensor
    @assert ndims(aug_img) == 3 "Each image should be a 3-tensor but got length $(ndims(aug_img))"
    @assert size(aug_img, 3) == CHANNELS "Expected 3rd entry to be color-channels but got dimension $(size(aug_img, 1))"
    @assert size(aug_img, 1) > 8 && size(aug_img, 2) > 8 "Expected the 1st and 3nd channels to be height and width, but got dimensions $(size(aug_img, 2)) and $(size(aug_img, 3))" # should be 224 or 256

    return aug_img, OneHotArrays.onehot(dataset.labels[i], 0:999)
end

# TODO(deprecation):
# Legacy ImageNet dataloader that performs JPEG decoding and augmentation
# at runtime. This should be replaced by `construct_dataloaders_chunked`
# (renamed to `construct_dataloaders`) once validated.
function construct_dataloaders(
        base_path::String, 
        train_batchsize, 
        val_batchsize, 
        image_size::Int; 
        dev=gpu_device(),
        shuffle=true,
        random_crop=true,
        )
    println("=> creating dataloaders.")

    @inline MaybeRandomResizeCrop(enabled::Bool, image_size::Int) =
        enabled ? RandomResizeCrop((image_size, image_size)) : identity

    train_augment =
        ScaleFixed((256, 256)) |>
        MaybeRandomResizeCrop(random_crop, image_size) |>
        PinOrigin() |>
        ImageToTensor() |>
        MakeColoredImage() |>
        ToEltype(Float32) |>
        Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_files, train_labels = load_imagenet1k(base_path, :train)

    train_dataset = FileDataset(train_files, train_labels, train_augment)

    val_augment =
        ScaleFixed((image_size, image_size)) |>
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

    # assert that the dataloader work as expected during initialization here
    x, y = first(train_dataloader)
    assert_hwcn(x)
    @assert size(x, 4) == length(onecold(y)) "There should be as many labels as number ob batches but evaluated: batch size n=$(size(x,4)), number of labels $(length(onecold(y)))"

    return dev(train_dataloader), dev(val_dataloader)
end

function imagenet_data(base_path::String, train_batchsize, val_batchsize, image_size::Int; dev=gpu_device())
    train_set, val_set = construct_dataloaders(base_path, train_batchsize, val_batchsize, image_size; dev=dev)
    # test_set = deepcopy(val_set)
    test_set = nothing

    return train_set, val_set, test_set
end

# ==============================================================================
# ImageNet preprocessing (offline, chunked)
# ==============================================================================

@inline function assert_hwcn(x; name::AbstractString="tensor")
    @assert ndims(x) == 4 "$name must be 4D (H,W,C,N), got ndims=$(ndims(x))"
    @assert size(x, 1) > 200 "$name should have height H=224 or 256, got H=$(size(x,1))"
    @assert size(x, 2) > 200 "$name should have width W=224 or 256, got W=$(size(x,2))"
    @assert size(x, 3) == CHANNELS "$name must have C=$CHANNELS as 3rd dimension, got size(x,2)=$(size(x,2))"
    return nothing
end

# TODO(refactor):
# These preprocessing utilities should eventually live in a separate
# file/module (e.g. imagenet_preprocessing.jl), not alongside dataloaders.
function make_preprocess_pipeline(image_size::Int)
    ScaleFixed((image_size, image_size)) |>
    PinOrigin() |>
    ImageToTensor() |>
    MakeColoredImage() |>
    ToEltype(Float32) |>
    Normalize(IMAGENET_MEAN, IMAGENET_STD)
end

function save_chunk(out_path, chunk_id, images, labels)
    filename = joinpath(out_path, @sprintf("chunk_%06d.jld2", chunk_id))
    @save filename images labels
end

# TODO(cleanup):
# - Remove temporary early-break used for testing.
# - Run once to build the full preprocessed ImageNet dataset.
# - Move this function into a dedicated preprocessing script or module.
function preprocess_split(
    base_path::String,
    out_path::String,
    split::Symbol;
    chunk_size::Int = 16,
    image_size::Int = 256,
)
    files, labels = load_imagenet1k(base_path, split)
    augment = make_preprocess_pipeline(image_size)

    mkpath(out_path)

    chunk_images = Array{Float32,4}(undef, image_size, image_size, CHANNELS, 0) # HWCN-tensor
    chunk_labels = Int64[]

    assert_hwcn(chunk_images; name="chunk of images")

    chunk_id = 1

    @showprogress for (i, (file, label)) in enumerate(zip(files, labels))

        # TODO: remove this after testing
        if i > 64 break end

        img = Image(FileIO.load(file))

        x = itemdata(DataAugmentation.apply(augment, img)) # (H,W,C)
        @assert ndims(x) == 3
        @assert size(x, 3) == CHANNELS
        @assert size(x, 1) > 100 && size(x, 2) > 100

        x = reshape(x, size(x)..., 1) # (H,W,C,1)
        @assert ndims(x) == 4
        @assert size(x, 3) == CHANNELS
        @assert size(x, 4) == 1

        chunk_images = cat(chunk_images, x; dims = 4)
        push!(chunk_labels, label)

        assert_hwcn(chunk_images; name="chunk of images")

        if size(chunk_images, 4) == chunk_size
            save_chunk(out_path, chunk_id, chunk_images, chunk_labels)
            chunk_id += 1
            chunk_images = Array{Float32,4}(undef, image_size, image_size, CHANNELS, 0)
            empty!(chunk_labels)
        end
    end

    if size(chunk_images, 4) > 0
        assert_hwcn(chunk_images; name="chunk of images")
        save_chunk(out_path, chunk_id, chunk_images, chunk_labels)
    end
end

# ==============================================================================
# Chunked ImageNet dataset and dataloader (intended future default)
# ==============================================================================

# TODO(finalize):
# ChunkedImageNet represents the preprocessed ImageNet dataset and is
# intended to become the default ImageNet dataset used during training.
struct ChunkedImageNet
    chunk_files::Vector{String}
    chunk_size::Int
    image_size::Int
    split::Symbol
end

struct ChunkedBatch
    images::Array{Float32,4}
    labels::Vector{Int}
end

struct DeviceDataLoader
    loader
    crop_size::Union{Nothing,Int}
    dev
end

# TODO(invariants):
# Dataset invariants (chunk size, shape, dtype) are checked once here.
# Consider making chunk_size a global constant or type parameter if stable.
function ChunkedImageNet(
    root::String,
    split::Symbol;
    chunk_size::Int,
    image_size::Int,
)
    dir = joinpath(root, string(split))
    files = sort(filter(f -> endswith(f, ".jld2"), readdir(dir; join=true)))
    @assert !isempty(files) "No chunks found in $dir"

    @load files[1] images labels

    assert_hwcn(images; name="images batch")
    @assert length(labels) == size(images, 4) "The saved labels should be integer vectors"

    return ChunkedImageNet(files, chunk_size, image_size, split)
end

Base.length(ds::ChunkedImageNet) = length(ds.chunk_files)

function Base.getindex(ds::ChunkedImageNet, i::Int)
    @load ds.chunk_files[i] images labels
    return ChunkedBatch(images, labels)
end

# TODO(performance):
# This collates chunk-level batches into a full batch.
# Assumes batch_size % chunk_size == 0.
function collate_chunks(batches::Vector{ChunkedBatch})
    images = cat((b.images for b in batches)...; dims=4)
    labels = vcat((b.labels for b in batches)...)
    assert_hwcn(images; name="iamges batch")
    return images, labels
end

# TODO(augmentation):
# Random cropping is applied at runtime and can run on CPU or GPU
# depending on `dev`. Evaluate whether this should remain here or
# move into the training loop.
function random_crop!(out, x, crop_size)

    assert_hwcn(x)

    H, W, _, _ = size(x)
    @assert H ≥ crop_size && W ≥ crop_size

    h0 = rand(0:(H - crop_size))
    w0 = rand(0:(W - crop_size))

    out .= @view x[h0+1:h0+crop_size, w0+1:w0+crop_size, :, :]
end

# TODO(abstraction):
# Thin wrapper that moves batches to `dev` and applies runtime augmentation.
# If this pattern generalizes, consider extracting a shared abstraction.

@inline function to_onehot(labels::Vector{Int})
    return OneHotArrays.onehotbatch(labels, 0:999)
end

function Base.iterate(dl::DeviceDataLoader, state...)
    res = iterate(dl.loader, state...)
    res === nothing && return nothing

    ((x, y), st) = res

    x = dl.dev(x)
    y = to_onehot(y) # onehot encode labels after loading at runtime
    y = dl.dev(y)

    assert_hwcn(x)

    if dl.crop_size !== nothing
        cropped = similar(
            x,
            dl.crop_size,
            dl.crop_size,
            size(x,3),
            size(x,4),
        )

        random_crop!(cropped, x, dl.crop_size)
        x = cropped
    end

    return (x, y), st
end

# TODO(rename):
# Once validated, this function should replace `construct_dataloaders`
# and be renamed accordingly.
function construct_dataloaders_chunked(
    root::String,
    train_batchsize::Int,
    val_batchsize::Int;
    chunk_size::Int = 16,
    train_image_size::Int = 256,
    val_image_size::Int = 224,
    crop_size::Union{Int, Nothing} = 224,
    dev = gpu_device(),
    workers::Int = 4,
    shuffle=true,
)
    @assert train_batchsize % chunk_size == 0
    @assert val_batchsize % chunk_size == 0

    train_ds = ChunkedImageNet(
        root, :train;
        chunk_size=chunk_size,
        image_size=train_image_size,
    )

    val_ds = ChunkedImageNet(
        root, :val;
        chunk_size=chunk_size,
        image_size=val_image_size,
    )

    train_loader = DataLoader(
        train_ds;
        batchsize = train_batchsize ÷ chunk_size,
        shuffle = shuffle,
        collate = collate_chunks,
        parallel = true,
    )

    val_loader = DataLoader(
        val_ds;
        batchsize = val_batchsize ÷ chunk_size,
        shuffle = false,
        collate = collate_chunks,
        parallel = true,
    )

    train_loader = DeviceDataLoader(train_loader, crop_size, dev)
    val_loader   = DeviceDataLoader(val_loader, nothing, dev)

    # assert that the dataloader work as expected during initialization here
    x, y = first(train_loader)
    assert_hwcn(x)
    @assert size(x, 4) != length(y) "The labels should be one-hot encoded"
    @assert size(x, 4) == length(onecold(y)) "There should be as many labels as number ob batches but evaluated: batch size n=$(size(x,4)), number of labels $(length(onecold(y)))"

    return train_loader, val_loader
end