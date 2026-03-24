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
    @assert size(x, 1) ≥ 32 "$name height too small: H=$(size(x,1))"
    @assert size(x, 2) ≥ 32 "$name width too small: W=$(size(x,2))"
    @assert size(x, 3) == CHANNELS "$name must have C=$CHANNELS as 3rd dimension, got size(x,3)=$(size(x,3))"
    return nothing
end

"""
    make_preprocess_pipeline(image_size::Int)

Create the deterministic preprocessing pipeline used for offline
ImageNet preprocessing.

Resizes images to `(image_size, image_size)` using ScaleFixed (squishes
non-square images). Random augmentations (random crop, horizontal flip)
are applied later at runtime by `DeviceDataLoader`.
"""
function make_preprocess_pipeline(image_size::Int)
    ScaleFixed((image_size, image_size)) |>
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
    perm = randperm(length(files))
    files = files[perm]
    labels = labels[perm]
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
    augment::Bool
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
    random_resized_crop!(out, x)

In-place random resized crop of HWCN tensor `x` into pre-allocated `out`.

Follows the PyTorch `RandomResizedCrop` algorithm: samples a random crop
area (8%–100% of image area) and log-uniform aspect ratio (0.75–1.33),
crops that region, then bilinear-resizes into `out`.

Falls back to a center crop of the full shorter side if no valid
parameters are found after 10 attempts.

Note: `NNlib.upsample_bilinear` allocates one intermediate tensor for
the resize result, which is then copied into `out` in-place.
"""
function random_resized_crop!(out, x)
    assert_hwcn(x)
    output_size = size(out, 1)
    H, W = size(x, 1), size(x, 2)
    area = Float32(H * W)

    crop_h, crop_w = H, W
    found = false
    for _ in 1:10
        target_area = area * (0.08f0 + rand(Float32) * 0.92f0)
        log_ratio   = log(0.75f0) + rand(Float32) * (log(1.3333334f0) - log(0.75f0))
        w = round(Int, sqrt(target_area * exp(log_ratio)))
        h = round(Int, sqrt(target_area / exp(log_ratio)))
        if 1 ≤ w ≤ W && 1 ≤ h ≤ H
            crop_h, crop_w = h, w
            found = true
            break
        end
    end

    if !found
        s  = min(H, W)
        h0 = (H - s) ÷ 2
        w0 = (W - s) ÷ 2
        out .= NNlib.upsample_bilinear(x[h0+1:h0+s, w0+1:w0+s, :, :]; size=(output_size, output_size))
        return out
    end

    h0 = rand(0:(H - crop_h))
    w0 = rand(0:(W - crop_w))
    out .= NNlib.upsample_bilinear(x[h0+1:h0+crop_h, w0+1:w0+crop_w, :, :]; size=(output_size, output_size))
    return out
end

"""
    center_crop(x, output_size)

Take the central `output_size × output_size` region of a HWCN tensor.
Used for deterministic validation augmentation.
"""
function center_crop(x, output_size::Int)
    H, W = size(x, 1), size(x, 2)
    @assert H ≥ output_size && W ≥ output_size
    h0 = (H - output_size) ÷ 2
    w0 = (W - output_size) ÷ 2
    return x[h0+1:h0+output_size, w0+1:w0+output_size, :, :]
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
- training: random resized crop (scale 0.08–1.0, ratio 0.75–1.33) + 50% hflip
- validation: deterministic center crop
"""
function Base.iterate(dl::DeviceDataLoader, state...)
    res = iterate(dl.loader, state...)
    res === nothing && return nothing

    ((x, y), st) = res

    x = dl.dev(x)
    assert_hwcn(x)
    y = dl.dev(to_onehot(y))

    if dl.crop_size !== nothing
        if dl.augment
            cropped = similar(x, dl.crop_size, dl.crop_size, size(x, 3), size(x, 4))
            random_resized_crop!(cropped, x)
            x = cropped
            if rand() < 0.5
                x = reverse(x, dims=2)
            end
        else
            x = center_crop(x, dl.crop_size)
        end
    end

    return (x, y), st
end

Base.length(dl::DeviceDataLoader) = length(dl.loader)

Base.IteratorSize(::Type{DeviceDataLoader}) = Base.HasLength()

Base.eltype(::Type{DeviceDataLoader}) = Tuple


"""
    construct_chunked_dataloaders(
        root,
        train_batchsize,
        val_batchsize;
        chunk_size=16,
        train_image_size=480,
        val_image_size=256,
        crop_size=224,
        dev=gpu_device(),
        workers=4,
        shuffle=true,
    )

Construct ImageNet dataloaders from offline preprocessed chunks.

Training: random resized crop (scale 0.08–1.0, ratio 0.75–1.33) +
random horizontal flip applied at runtime via `DeviceDataLoader`.

Validation: deterministic center crop to `crop_size` at runtime.

Train chunks should be preprocessed at `train_image_size=480` and
val chunks at `val_image_size=256` using `ScaleFixed`.
"""
function construct_chunked_dataloaders(
    root::String,
    train_batchsize::Int,
    val_batchsize::Int;
    chunk_size::Int = 16,
    train_image_size::Int = 480,
    val_image_size::Int = 256,
    crop_size::Union{Int, Nothing} = 224,
    dev = gpu_device(),
    workers::Int = 4,
    shuffle=true,
)
    @assert train_batchsize % chunk_size == 0 """
        train_batchsize ($train_batchsize) must be divisible by chunk_size ($chunk_size).
        The chunked dataloader loads chunk_size images per file; batch size must be a 
        multiple of this. Either set train_batchsize to a multiple of $chunk_size, 
        or re-preprocess ImageNet with a different chunk_size.
    """
    @assert val_batchsize % chunk_size == 0 """
        val_batchsize ($val_batchsize) must be divisible by chunk_size ($chunk_size).
    """

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

    train_loader = DeviceDataLoader(train_loader, crop_size, dev, true)
    val_loader   = DeviceDataLoader(val_loader, crop_size, dev, false)

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
    train_batchsize::Int,
    val_batchsize::Int;
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