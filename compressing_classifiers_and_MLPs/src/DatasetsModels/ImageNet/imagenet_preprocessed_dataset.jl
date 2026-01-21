# ==============================================================================
# ImageNet preprocessing (offline, chunked)
# ==============================================================================

"""
    assert_hwcn(x; name="tensor")

Assert that a tensor follows the ImageNet HWCN memory layout.

# Expected layout
- Dimensions: `(H, W, C, N)`
- `H, W`: height and width (typically 224 or 256)
- `C`: number of channels (`CHANNELS`)
- `N`: batch dimension

Throws an assertion error if the layout or dimensions are invalid.
"""
@inline function assert_hwcn(x; name::AbstractString="tensor")
    @assert ndims(x) == 4 "$name must be 4D (H,W,C,N), got ndims=$(ndims(x))"
    @assert size(x, 1) > 200 "$name should have height H=224 or 256, got H=$(size(x,1))"
    @assert size(x, 2) > 200 "$name should have width W=224 or 256, got W=$(size(x,2))"
    @assert size(x, 3) == CHANNELS "$name must have C=$CHANNELS as 3rd dimension, got size(x,3)=$(size(x,3))"
    return nothing
end

"""
    make_preprocess_pipeline(image_size::Int)

Create the deterministic preprocessing pipeline used for offline
ImageNet preprocessing.

This pipeline:
- Resizes images to `(image_size, image_size)`
- Converts grayscale images to RGB
- Converts to `Float32`
- Normalizes using ImageNet statistics

Random augmentations are intentionally excluded and are applied later
at runtime.
"""
function make_preprocess_pipeline(image_size::Int)
    ScaleFixed((image_size, image_size)) |>
    PinOrigin() |>
    ImageToTensor() |>
    MakeColoredImage() |>
    ToEltype(Float32) |>
    Normalize(IMAGENET_MEAN, IMAGENET_STD)
end

"""
    save_chunk(out_path, chunk_id, images, labels)

Save a preprocessed ImageNet chunk to disk in `.jld2` format.

Each chunk contains:
- `images`: Float32 tensor of shape `(H, W, C, N)`
- `labels`: Vector of integer class labels

Chunks are named `chunk_XXXXXX.jld2`.
"""
function save_chunk(out_path, chunk_id, images, labels)
    filename = joinpath(out_path, @sprintf("chunk_%06d.jld2", chunk_id))
    @save filename images labels
end

"""
    preprocess_split(
        base_path::String,
        out_path::String,
        split::Symbol;
        chunk_size::Int = 16,
        image_size::Int = 256,
    )

Offline preprocessing of ImageNet images into fixed-size chunks.

# Arguments
- `base_path`: Root ImageNet directory (`train/`, `val/`).
- `out_path`: Output directory for chunk files.
- `split`: Either `:train` or `:val`.
- `chunk_size`: Number of images per saved chunk.
- `image_size`: Spatial resolution used during preprocessing.

# Output
Writes `.jld2` files containing `(images, labels)` pairs to `out_path`.

This function performs:
- JPEG decoding
- Deterministic resizing and normalization
- Chunking into fixed-size HWCN tensors

Intended to be run once as an offline preprocessing step.
"""
function preprocess_split(
    base_path::String,
    out_path::String,
    split::Symbol;
    chunk_size::Int = 16,
    image_size::Int = 256,
    debug=false
)
    files, labels = load_imagenet1k(base_path, split)
    augment = make_preprocess_pipeline(image_size)

    mkpath(out_path)

    chunk_images = Array{Float32,4}(undef, image_size, image_size, CHANNELS, 0) # HWCN-tensor
    chunk_labels = Int64[]

    assert_hwcn(chunk_images; name="chunk of images")

    chunk_id = 1

    @showprogress for (i, (file, label)) in enumerate(zip(files, labels))

        if debug && i > 128
            break
        end

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
# Chunked ImageNet dataset and dataloader
# ==============================================================================

"""
Dataset representing ImageNet stored as preprocessed chunks on disk.
"""
struct ChunkedImageNet
    chunk_files::Vector{String}
    chunk_size::Int
    image_size::Int
    split::Symbol
end

"""
Single batch loaded from a chunk file.
"""
struct ChunkedBatch
    images::Array{Float32,4}
    labels::Vector{Int}
end

"""
Wrapper around a `DataLoader` that applies device transfer, optional
runtime cropping, and one-hot encoding.
"""
struct DeviceDataLoader
    loader
    crop_size::Union{Nothing,Int}
    dev
end

"""
    ChunkedImageNet(root, split; chunk_size, image_size)

Create a `ChunkedImageNet` dataset from preprocessed chunk files.

Validates the first chunk to ensure correct tensor layout.
"""
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
    @assert length(labels) == size(images, 4)

    return ChunkedImageNet(files, chunk_size, image_size, split)
end

"""
Return the number of chunks in the dataset.
"""
Base.length(ds::ChunkedImageNet) = length(ds.chunk_files)

"""
Load the `i`-th chunk from disk.
"""
function Base.getindex(ds::ChunkedImageNet, i::Int)
    @load ds.chunk_files[i] images labels
    return ChunkedBatch(images, labels)
end

"""
    collate_chunks(batches)

Collate multiple `ChunkedBatch` objects into a single batch by
concatenating along the batch dimension.
"""
function collate_chunks(batches::Vector{ChunkedBatch})
    images = cat((b.images for b in batches)...; dims=4)
    labels = vcat((b.labels for b in batches)...)
    assert_hwcn(images; name="images batch")
    return images, labels
end

"""
    random_crop!(out, x, crop_size)

Apply a random spatial crop to a HWCN tensor.

This operation is performed at runtime and can run on CPU or GPU,
depending on the device of `x`.
"""
function random_crop!(out, x, crop_size)
    assert_hwcn(x)

    H, W, _, _ = size(x)
    @assert H ≥ crop_size && W ≥ crop_size

    h0 = rand(0:(H - crop_size))
    w0 = rand(0:(W - crop_size))

    out .= @view x[h0+1:h0+crop_size, w0+1:w0+crop_size, :, :]
end

"""
Convert integer labels to one-hot encoding over ImageNet classes.
"""
@inline function to_onehot(labels::Vector{Int})
    return OneHotArrays.onehotbatch(labels, 0:999)
end

"""
Iterator implementation for `DeviceDataLoader`.

Applies:
- device transfer
- one-hot encoding of labels
- optional random cropping
"""
function Base.iterate(dl::DeviceDataLoader, state...)
    res = iterate(dl.loader, state...)
    res === nothing && return nothing

    ((x, y), st) = res

    x = dl.dev(x)
    y = dl.dev(to_onehot(y)) # onehot encode labels after loading at runtime

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

"""
    construct_chunked_dataloaders(
        root,
        train_batchsize,
        val_batchsize;
        chunk_size=16,
        train_image_size=256,
        val_image_size=224,
        crop_size=224,
        dev=gpu_device(),
        workers=4,
        shuffle=true,
    )

Construct ImageNet dataloaders from offline preprocessed chunks.

Random cropping is applied at runtime for the training loader.
"""
function construct_chunked_dataloaders(
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
    @assert size(x, 4) == length(onecold(y))

    return train_loader, val_loader
end

"""
    imagenet_preprocessed_data(root, train_batchsize, val_batchsize; kwargs...)

High-level entry point for using **offline preprocessed ImageNet data**.

Returns `(train, val, test)` where `test` is always `nothing`.
"""
function imagenet_preprocessed_data(
    root::String,
    train_batchsize,
    val_batchsize;
    kwargs...
)
    train, val = construct_chunked_dataloaders(
        root,
        train_batchsize,
        val_batchsize;
        kwargs...
    )
    test = nothing
    return train, val, test
end
