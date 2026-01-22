"""
Prepare ImageNet for training.

This script:
1. Validates the ImageNet directory structure
2. Checks whether preprocessed (chunked) datasets already exist
3. Rebuilds the preprocessed dataset if missing or incomplete

Expected config.toml entries:
[paths]
imagenet_path = "/path/to/imagenet"
imagenet_preprocessed_path = "/path/to/imagenet_preprocessed"

Adapted to the ImageNet pipeline in `DatasetsModels.ImageNet`.
"""

# ==============================================================================
# Environment & imports
# ==============================================================================

using Pkg
Pkg.activate(".")

using TOML
using Printf
using Statistics
using ProgressMeter
using Lux

using CompressingClassifiersMLPs.DatasetsModels.ImageNet:
    preprocess_split,
    IMAGENET_MEAN,
    IMAGENET_STD

# ==============================================================================
# Configuration
# ==============================================================================

const TRAIN_IMAGES = 1_281_167
const VAL_IMAGES   = 50_000

const TRAIN_IMAGE_SIZE = 256
const VAL_IMAGE_SIZE   = 224
const CHUNK_SIZE       = 32

const EXPECTED_SYNSETS = 1000

# ==============================================================================
# Utilities
# ==============================================================================

function warn_threads()
    println("Using $(Threads.nthreads()) Julia threads")
    if Threads.nthreads() < 4
        @warn """
        You are running with fewer than 4 Julia threads.
        Image decoding and preprocessing will be significantly slower.

        Consider launching Julia with:
            julia -t auto
        """
    end
end

function check_imagenet_root(root::String)
    @assert isdir(root) "ImageNet root not found: $root"

    for split in ("train", "val")
        split_dir = joinpath(root, split)
        @assert isdir(split_dir) "Missing directory: $split_dir"

        synsets = filter(isdir, readdir(split_dir; join=true))
        @assert length(synsets) == EXPECTED_SYNSETS """
        Expected $EXPECTED_SYNSETS synset directories in $split_dir,
        found $(length(synsets))
        """
    end
end

function expected_chunks(n_images::Int, chunk_size::Int)
    return ceil(Int, n_images / chunk_size)
end

function check_preprocessed_split(
    dir::String;
    expected_images::Int,
    chunk_size::Int,
)
    if !isdir(dir)
        return false
    end

    files = sort(filter(f -> endswith(f, ".jld2"), readdir(dir; join=true)))
    isempty(files) && return false

    n_chunks_expected = expected_chunks(expected_images, chunk_size)

    if length(files) != n_chunks_expected
        @warn """
        Found $(length(files)) chunks in $dir,
        expected $n_chunks_expected.
        """
        return false
    end

    return true
end

# ==============================================================================
# Main
# ==============================================================================

function main()
    warn_threads()

    config_path = joinpath(@__DIR__, "..", "config.toml")
    @assert isfile(config_path) "config.toml not found at $config_path"

    config = TOML.parsefile(config_path)

    imagenet_root = config["paths"]["imagenet_path"]
    out_root      = config["paths"]["imagenet_preprocessed_path"]

    println("ImageNet root:       ", imagenet_root)
    println("Preprocessed output: ", out_root)

    # --------------------------------------------------------------------------
    # Validate ImageNet directory
    # --------------------------------------------------------------------------
    println("\nChecking ImageNet directory structure...")
    check_imagenet_root(imagenet_root)
    println("✅ ImageNet directory looks valid")

    # --------------------------------------------------------------------------
    # Check preprocessed datasets
    # --------------------------------------------------------------------------
    train_out = joinpath(out_root, "train")
    val_out   = joinpath(out_root, "val")

    println("\nChecking preprocessed dataset...")

    train_ok = check_preprocessed_split(
        train_out;
        expected_images=TRAIN_IMAGES,
        chunk_size=CHUNK_SIZE,
    )

    val_ok = check_preprocessed_split(
        val_out;
        expected_images=VAL_IMAGES,
        chunk_size=CHUNK_SIZE,
    )

    if train_ok && val_ok
        println("✅ Preprocessed ImageNet already exists and looks complete")
        return
    end

    # --------------------------------------------------------------------------
    # Rebuild preprocessing
    # --------------------------------------------------------------------------
    println("\n⚠️  Preprocessed dataset missing or incomplete")
    println("Rebuilding ImageNet preprocessing...\n")

    mkpath(out_root)

    println("Preprocessing TRAIN split...")
    preprocess_split(
        imagenet_root,
        train_out,
        :train;
        chunk_size=CHUNK_SIZE,
        image_size=TRAIN_IMAGE_SIZE,
    )

    println("\nPreprocessing VAL split...")
    preprocess_split(
        imagenet_root,
        val_out,
        :val;
        chunk_size=CHUNK_SIZE,
        image_size=VAL_IMAGE_SIZE,
    )

    println("\n✅ ImageNet preprocessing completed successfully")
end

# ==============================================================================
# Entrypoint
# ==============================================================================

main()

