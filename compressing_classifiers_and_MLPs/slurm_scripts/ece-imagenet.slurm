#!/bin/bash
#SBATCH --job-name=ece-imagenet
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Run: sbatch -p gpu slurm_scripts/ece-imagenet.slurm

cd "/projects/latentSimplicity/code/efficient-compression/compressing_classifiers_and_MLPs"
nvidia-smi

stdbuf -oL -eL julia --threads=8 "run_ECE_benchmark.jl" 1>&2
