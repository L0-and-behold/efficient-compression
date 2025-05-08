# Compressing Wikipedia
Codebase for the experiments from our paper regarding language models and Wikipedia dataset compression.

## Table of Contents
- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Download the Dataset](#download-the-dataset)
  - [Run the Experiment](#run-the-experiment)
- [Experiment Parameters and Setup](#experiment-parameters-and-setup)
  - [Regularization Parameters](#regularization-parameters)
  - [Model Configuration](#model-configuration)
  - [Training Process Parameters](#training-process-parameters)
  - [Pruning Parameters](#pruning-parameters)
  - [Model Loading Parameters](#model-loading-parameters)
  - [General Training Parameters](#general-training-parameters)
  - [Metrics Calculation](#metrics-calculation)
- [Standard Dataset Configurations](#standard-dataset-configurations)
  - [16MB Dataset](#16mb-dataset)
  - [50MB Dataset](#50mb-dataset)
  - [300MB Dataset](#300mb-dataset)
  - [10GB Dataset](#10gb-dataset)
  - [Common Default Parameters](#common-default-parameters)
- [Running with Distributed Data Parallel (DDP)](#running-with-distributed-data-parallel-ddp)

## Quick Start

### Environment Setup

We assume that we work with Anaconda or Miniconda.

1. Create a new environment
```shell
conda create -n wikipedia-experiments
conda activate wikipedia-experiments
```

2. Install PyTorch with GPU support according to the GPU at hand
   (see https://pytorch.org/get-started/locally/)
   For our machine, the command reads:
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install the other dependencies
```shell
conda install pip
pip install -r requirements.txt
```

### Download the Dataset

Run:
```shell
cd path-to-project/compressing_wikipedia/
python ./src/Datasets/Wiki40BDataset.py
```
This will download the TensorFlow dataset Wiki40B/english, restructure it as a continuous byte string, and save it for use with PyTorch.

### Run the Experiment

You can modify `train.py` directly to set your parameters and then run.

```shell
cd path-to-project/compressing_wikipedia/
python train.py
```

## Experiment Parameters and Setup

After the experiment finishes, the results are saved into a folder including a `runs.csv` file containing parameters and calculated metrics as well as an artifact folder with the trained model parameters and loss curves.

Set the path for results storage in `train.py` by changing:
```python
path_to_database = os.path.join(os.getcwd(), "experiment-results")
experiment_name = "example-experiment"
```

The parametrs controlling the training can be set by modifying the `args` dictionary in `train.py`. The following documents what each of these parameters controlls.

### Regularization Parameters

| Parameter | Description |
|-----------|-------------|
| `alpha` | Regularization strength for DRR or RL1 procedures |
| `pmmp` | Set to True when using the PMMP procedure, False otherwise |
| `initial_p_value` | Initial parameter value for PMMP procedure (probability threshold) |
| `beta` | Parameter for DRR (Deterministic Reparameterization with Regularization) |

### Model Configuration

| Parameter | Description |
|-----------|-------------|
| `transformer_config` | Model size selection: "transformer200k", "transformer800k", or "transformer4.1m" (see TransformerConfig.py) |
| `training_method` | Regularization procedure selection: `rl1_procedure`, `pmmp_procedure`, `drr_procedure`, or `vanilla_procedure` function|

### Training Process Parameters

| Parameter | Description |
|-----------|-------------|
| `train_only_on_leading_tokens` | Training set size in number of tokens/bytes. Set to False for full dataset or integer for subset |
| `epochs_prelude` | Number of epochs to train before starting main regularized training |
| `epochs` | Number of epochs for main regularized training |
| `epochs_fine_tuning` | Number of epochs for fine-tuning with reduced transformer (unregularized training) |
| `stop_epoch_at_batch_prelude` | If integer, limits effective training set for prelude epochs to specified number of batches |
| `stop_epoch_at_batch` | If integer, limits effective training set for main training to specified number of batches |
| `stop_epoch_at_batch_fine_tuning` | If integer, limits effective training set for fine-tuning to specified number of batches |
| `batch_size` | Number of sequences processed at once (must be ≥ number of GPUs) |

### Pruning Parameters

| Parameter | Description |
|-----------|-------------|
| `do_pruning` | Whether to perform pruning of model weights |
| `first_pruning_after` | Start pruning during regularized training after this many epochs |
| `prune_every` | Frequency of pruning steps in epochs |

### Model Loading Parameters

| Parameter | Description |
|-----------|-------------|
| `use_pretrained_model` | Whether to load a pre-trained model and continue training |
| `use_model_from_experiment` | Name of experiment to load model from (if using pre-trained model) |
| `use_model_from_run` | Run ID to load model from (check runs.csv of the specified experiment) |
| `elapsed_epochs` | Number of completed epochs + planned epochs (tracking total training time) |

### General Training Parameters

| Parameter | Description |
|-----------|-------------|
| `learning_rate` | Initial learning rate for Adam optimizer |
| `seed` | Random seed for reproducibility |
| `tolerated_relative_loss_increase` | Threshold parameter for Threshold Adaptive Mask Determination (TAMADE) |
| `steps_per_chunk` | Number of gradient update steps to perform on each provided minibatch/chunk |
| `log_every` | Log loss every N update steps (set to 1 for online description length calculation) |
| `checkpoint_time` | Save checkpoint every N seconds |
| `max_runtime` | Maximum runtime in seconds before stopping |

### Metrics Calculation

| Setting | Description |
|---------|-------------|
| `calculate_test_loss` | Whether to calculate and log loss on test set |
| `calculate_train_loss` | Whether to calculate and log loss on training set |
| `calculate_model_byte_size` | Whether to calculate and log model size in bytes |
| `calculate_non_zero_params` | Whether to calculate and log count of non-zero parameters |
| `calculate_on_line_code_length` | Whether to calculate online description length (requires log_every=1) |
| `debug` | Whether to enable debug mode with additional logging |

## Standard Dataset Configurations

We provide standard configurations for different dataset sizes used in our experiments.

### 16MB Dataset

```python
# Training Process Parameters
"train_only_on_leading_tokens": 16384000,  # 16.4MB
"epochs_prelude": 150,
"epochs": 200,
"elapsed_epochs": 200,
"epochs_fine_tuning": 100,
"stop_epoch_at_batch_prelude": False,
"stop_epoch_at_batch": False,
"stop_epoch_at_batch_fine_tuning": False,

# Pruning Parameters
"do_pruning": True,
"first_pruning_after": 150,  # epochs
"prune_every": 25,  # epochs
```

### 50MB Dataset

```python
# Training Process Parameters
"train_only_on_leading_tokens": 50003968,  # 50MB
"epochs_prelude": 100,
"epochs": 150, 
"elapsed_epochs": 150,
"epochs_fine_tuning": 75,
"stop_epoch_at_batch_prelude": False,
"stop_epoch_at_batch": False,
"stop_epoch_at_batch_fine_tuning": False,

# Pruning Parameters
"do_pruning": True,
"first_pruning_after": 100,  # epochs
"prune_every": 10,  # epochs
```

### 300MB Dataset

```python
# Training Process Parameters
"train_only_on_leading_tokens": 299991040,  # 300MB
"epochs_prelude": 20,
"epochs": 30,
"elapsed_epochs": 30,
"epochs_fine_tuning": 10,
"stop_epoch_at_batch_prelude": False,
"stop_epoch_at_batch": False,
"stop_epoch_at_batch_fine_tuning": False,

# Pruning Parameters
"do_pruning": True,
"first_pruning_after": 20,  # epochs
"prune_every": 1,  # epochs
```

### 9.3GB Dataset

```python
# Training Process Parameters
"train_only_on_leading_tokens": False,  # use full dataset
"epochs_prelude": 0,
"epochs": 1,
"elapsed_epochs": 1,
"epochs_fine_tuning": 1,
"stop_epoch_at_batch_prelude": False,
"stop_epoch_at_batch": False,
"stop_epoch_at_batch_fine_tuning": 50000,  # limit to 50k batches

# Pruning Parameters
"do_pruning": True,
"first_pruning_after": 1,  # epochs
"prune_every": 1,  # epochs
```

### Common Default Parameters

These parameters are used across all dataset configurations:

```python
# Model Configuration
"transformer_config": "transformer4.1m",
"use_pretrained_model": False,

# Procedure Arguments
"beta": 5.0
"initial_p_value": 1.0

# General Training Parameters
"batch_size": 8,  # (except 9.3GB dataset which uses 4)
"learning_rate": 1e-5,
"seed": 858,
"tolerated_relative_loss_increase": 0.01,
"steps_per_chunk": 1,
"log_every": 1,
"checkpoint_time": 83000,  # ~23 hours
"max_runtime": 86000,  # ~24 hours

# Model Evaluation Settings
"minimal_context_window": 1500,
```


## Running with Distributed Data Parallel (DDP)

To run the experiment on multiple GPUs using data distributed training, assuming the SLURM system is used for job submission and 8 GPUs are being utilized, submit a job similar to:

```slurm
#!/bin/bash -l

#SBATCH --open-mode=append
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4

#SBATCH --job-name=wikipedia-experiment

cd ~/path-to-project/compressing_wikipedia/

eval "$(conda shell.bash hook)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wikipedia-experiments

# export environment variables to python script
export PYTHONUNBUFFERED=1
export MASTER_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=29500

echo "MASTER_IP: $MASTER_IP"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_PORT: $MASTER_PORT"

# run python script
srun --ntasks=$WORLD_SIZE --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    bash -c 'export RANK=$SLURM_PROCID; \
    echo "RANK: $RANK"; \
    python -u ~/path-to-project/compressing_wikipedia/train.py'
```

**Important Note**: The number of GPUs cannot be higher than the batch size. This is because training is done in a close-to-online setting, which does not allow for finer breakup of data than one sequence of tok
