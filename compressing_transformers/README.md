# Compressing Transformers
Codebase for the experiments from our paper regarding language models and Wikipedia dataset compression.

## Table of Contents
- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Download the Dataset](#download-the-dataset)
  - [Run the Experiment](#run-the-experiment)
  - [Conventional Compressor Benchmarks](#conventional-compressor-benchmarks)
  - [Analysis Scripts](#analysis-scripts)
    - [IUC — Information Under Curve](#iuc--information-under-curve)
    - [plots.py — Training Plots](#plotspy--training-plots)
- [Experiment Parameters and Setup](#experiment-parameters-and-setup)
  - [Regularization Parameters](#regularization-parameters)
  - [Model Configuration](#model-configuration)
  - [Training Process Parameters](#training-process-parameters)
  - [Pruning Parameters](#pruning-parameters)
  - [Model Loading Parameters](#model-loading-parameters)
  - [General Training Parameters](#general-training-parameters)
  - [Metrics Calculation](#metrics-calculation)
- [Standard Dataset Configurations](#standard-dataset-configurations)
  - [300 MB Dataset](#300-mb-dataset)
  - [1.23 GB Dataset](#123-gb-dataset)
  - [6.16 GB Dataset](#616-gb-dataset)
  - [Common Default Parameters](#common-default-parameters)
- [SLURM arrays with Distributed Data Parallel (DDP)](#slurm-arrays-with-distributed-data-parallel-ddp)

## Quick Start

### Environment Setup

The code assumes, that a CUDA gpu-device is present.

1. Make sure that we are working in the correct directory
```shell
cd path/to/efficient-compression/compressing_transformers
```

2. Create a new environment
In the following, we assume usage of Anaconda or Miniconda or Miniforge (see [Installing Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [Installing Miniforge](https://conda-forge.org/download/)). After installation create an environment as follows:
```shell
conda create -n wikipedia-experiments python=3.11
conda activate wikipedia-experiments
```

3. Install PyTorch with GPU support according to the GPU at hand
   (see https://pytorch.org/get-started/locally/)
   For our machine, the command reads:
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. Install the other dependencies
```shell
conda install pip
pip install -r requirements.txt
```

### Download the Dataset

Run:
```shell
python ./src/Datasets/Wiki40BDataset.py
```
This will download the TensorFlow dataset Wiki40B/english, restructure it as a continuous byte string, and save it for use with PyTorch.

### Run the Experiment

You can modify `train.py` directly to set your parameters and then run.

```shell
python train.py
```

### Conventional Compressor Benchmarks

To benchmark LZMA2 and Zstandard compression on the dataset run:
```shell
python conventional_compressors.py
```

This looks for `processed_wiki_dataset_512.pt` in the current directory (matching the default sequence length of 512). For a different sequence length or file location:
```shell
python conventional_compressors.py --dataset /path/to/processed_wiki_dataset_1024.pt --seq-len 1024
# or point to a directory and specify the sequence length:
python conventional_compressors.py --dataset /path/to/dir --seq-len 1024
```

Both compressors are evaluated on the three standard dataset sizes (0.300 GB, 1.232 GB, 6.160 GB) using a chunk size equal to the sequence length, matching the online coding setting used for transformer evaluation. Results are written to `output/conventional_compressors_benchmark.out`.

Additional options:
```shell
python conventional_compressors.py --sizes 299991040 1232000000 6159990784
python conventional_compressors.py --workers 8
```

## Analysis Scripts

### IUC — Information Under Curve

`IUC.py` computes the coding length of a training run by integrating the per-iteration loss curve over the first epoch.

The input is the `train_loss.csv` artifact produced during training (found under `artifacts/run-<id>/train_loss.csv`), renamed to reflect the dataset size in bytes (e.g. `299991040.csv`). It must have exactly two columns: iteration number (starting at 1) and training loss in nats.

```shell
python IUC.py -i input/299991040.csv
```

Output is printed to stdout and saved to `output/<stem>_<timestamp>.out`.

### plots.py — Training Plots

`plots.py` generates two PDF plots from a `runs.csv` of transformer experiment results.

```shell
python plots.py -i input/runs.csv
python plots.py -i input/runs.csv --linear-x   # linear x-axis for Plot 1
```

**Plot 1 — Loss vs. Model Size**: scatter plot of mean test loss vs. model byte size (log-x by default, or linear with `--linear-x`), with equipotential iso-lines of constant description length (DL):

$$\text{DL} = \text{model\_bytes} + \frac{\text{loss} \times \text{dataset\_size}}{\ln 2 \times 8}$$

**Plot 2 — Description Length vs. α**: DL vs. regularization parameter α (log-x), with reference lines for the vanilla baseline and raw dataset size.

Required columns: `training_procedure`, `model_byte_size`, `mean_test_loss`, `alpha`, `train_only_on_leading_tokens`, `transformer_config`.

Example files are provided in `src/mdl_analysis/example-input/`:

- `test.csv` — example `runs.csv` with made-up data, for testing `plots.py`
- `299991040.csv` — example `train_loss.csv` (renamed to dataset size), for testing `IUC.py`

```shell
python plots.py -i src/mdl_analysis/example-input/test.csv
python IUC.py -i src/mdl_analysis/example-input/299991040.csv
```

Outputs are saved to `output/` (PDFs and `.out` report).

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
| `initial_p_value` | Initial `p` parameter value for PMMP procedure (probability threshold) |
| `initial_u_value` | Initial `u` parameter value for PMMP procedure (constraint enforcement strength) |
| `beta` | Parameter for DRR (Deterministic Reparameterization with Regularization) |

### Model Configuration

| Parameter | Description |
|-----------|-------------|
| `training_procedure` | Regularization procedure selection: `rl1_procedure`, `pmmp_procedure`, `drr_procedure`, or `vanilla_procedure` function|
| `transformer_config` | Model size selection: Viable model names are listed in TransformerConfig.py. For example, "transformer200k" or "t307_38p". The notation tX_Y denotes a transformer with X parameters and roughly Y GB of peak VRAM usage per GPU. If there is a 'p' behind Y, then Y denotes peak VRAM usage requirements for PMMP (which has more parameters than the other methods). |

### Optimizer Parameters

| Parameter | Description |
|-----------|-------------|
| `learning_rate` | Initial learning rate for optimizer |
| `AdamW_betas` | The beta parameters for the AdamW optimizer (typical choices include (0.9, 0.95) or (0.9, 0.999)) |
| `warmup_steps` | Warmup increases the learning rate linearly to `learning_rate` in `warmup_steps` steps |
| `weight_decay` | Weight decay applies L2 regularization to all parameters except biases, LayerNorm-weights and `u` and `p` parameters of PMMP |


### Training Process Parameters

| Parameter | Description |
|-----------|-------------|
| `iterations_per_epoch` | The number of iterations or batches which are processed per epoch. Each iteration, `batch_size` many `seq_length` long batches are processed. Therefore, make sure the dataset contains at least `args["iterations_per_epoch"]*batch_size*seq_length` many tokens (where `seq_length` is the context window of the model). |
| `tokens_per_epoch` | (False or int). If not False, then ["iterations_per_epoch"] is overwritten and set equal to `int(args["tokens_per_epoch"] / batch_size / seq_length)`, where `seq_length` is the context window of the model. Make sure `args["tokens_per_epoch"]` is not bigger than the number of tokens in your dataset. This parameter can also be used to make the training token number per epoch equal to `N` times the number of parameters of the model. |
| `epochs_prelude` | Number of epochs to train before starting main regularized training |
| `epochs` | Number of epochs for main regularized training |
| `epochs_fine_tuning` | Number of epochs for fine-tuning with reduced transformer (unregularized training) |
| `train_only_on_leading_tokens` | Derived value (not directly settable). Computed by TrainFunctions as `iterations_per_epoch × batch_size × seq_length`; reflects the total number of tokens trained per epoch and is logged in `runs.csv` for reference. |
| `stop_epoch_at_batch_prelude` | If integer, limits effective training set for prelude epochs to specified number of batches |
| `stop_epoch_at_batch` | If integer, limits effective training set for main training to specified number of batches |
| `stop_epoch_at_batch_fine_tuning` | If integer, limits effective training set for fine-tuning to specified number of batches |

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
| `only_process_every_nth_batch_when_calculating_train_loss` | Sample every Nth batch when computing training loss (default: 5; set to 1 for full evaluation) |
| `only_process_every_nth_batch_when_calculating_test_loss` | Sample every Nth batch when computing test loss (default: 1) |

## Standard Dataset Configurations

We provide standard configurations for different dataset sizes used in our experiments.

Note that each iteration, `batch_size` many `seq_length` (context window) long batches are processed. The number of tokens per batch thus equals `batch_size * seq_length`.

Consequently, if we want to train on a dataset whose token number is `n` times larger than the number `p` of parameters of the transformer, we have to set the `iterations_per_epoch` to `int(n * p / batch_size / seq_length)`.

From this iteration number, the size of the dataset is automatically computed. Since each token is encoded by one byte, setting iterations this way will result in a dataset size around `n * p` bytes.

We conducted experiments with `batch_size=128` and `seq_length=512`, using a transformer with roughly `308 million` parameters and set `iterations_per_epoch` accordingly in the configurations below.

### 300 MB Dataset

```python
# Training Process Parameters
"iterations_per_epoch": int(300000000 / 128 / 512),  # 300 MB. Number of tokens roughly correspond to the number of parameters of the transformer.
"epochs": 20,
"epochs_fine_tuning": 3,
"stop_epoch_at_batch_fine_tuning": False,

# Pruning Parameters
"do_pruning": True,
"first_pruning_after": 20,  # epochs
"prune_every": 20,  # epochs
```

### 1.23 GB Dataset

```python
# Training Process Parameters
"iterations_per_epoch": int(4 * 308 * 10**6 / 128 / 512),  # 1.23 GB. Number of tokens 4 times higher than the number of parameters of the transformer.
"epochs": 4,
"epochs_fine_tuning": 1,
"stop_epoch_at_batch_fine_tuning": False,

# Pruning Parameters
"do_pruning": True,
"first_pruning_after": 4,  # epochs
"prune_every": 4,  # epochs
```

### 6.16 GB Dataset

```python
# Training Process Parameters
"iterations_per_epoch": int(20 * 308 * 10**6 / 128 / 512),  # 1.23 GB. Number of tokens 20 times higher than the number of parameters of the transformer.
"epochs": 1, # train for a single epoch.
"epochs_fine_tuning": 1,
"stop_epoch_at_batch_fine_tuning": int(0.15*args["iterations_per_epoch"]), # stop after 15 percent of the finetuning epoch because the epoch is very long here.

# Pruning Parameters
"do_pruning": True,
"first_pruning_after": 1,  # epochs
"prune_every": 1,  # epochs
```

The number of epochs is chosen such that the number of iterations stays roughly the same across the different datasets.

### Common Default Parameters

These parameters are used across all dataset configurations:

```python
# Model Configuration
"transformer_config": "t307_38p",

# Procedure Arguments
"initial_p_value" = 1.0  # Initial p value for PMMP method
"initial_u_value" = 0.0  # Initial u value for PMMP method

### Optimizer Parameters - BEGIN
"learning_rate" = 3e-4   # Initial learning rate for optimizer
"AdamW_betas" = (0.9, 0.95)  # The beta parameters for the AdamW optimizer (typical choices include (0.9, 0.95) or (0.9, 0.999))
"warmup_steps" = 2000 # Warmup increases the learning rate linearly to `learning_rate` in `warmup_steps` steps

### Model Loading Parameters - BEGIN
"use_pretrained_model" = False  # Whether to use a pretrained model

# General Training Parameters
"seed": 858,
"tolerated_relative_loss_increase": 0.02,

### Metrics Calculation Parameters - BEGIN
# (evaluated after training)
other_settings = {
    "calculate_test_loss": True,  # Whether to calculate mean loss on test set
}
```

Other default parameters are given in `train.py`.

## SLURM arrays with Distributed Data Parallel (DDP)

We briefly comment on how to use SLURM arrays to submit multiple runs to evaluate multiple parameter values in parallel on a cluster that supports this.

We assume the following folder structure:
```
compressing_transformers/
├── src/
│   └── ...
└── experiment_scripts/
│   ├── train_config.py
│   └── drr_param_sweep.csv
└── slurm_scripts/
│   ├── submit_experiment.sh
│   └── launch_experiment.sh
└── experiment-results/
│   └── ...
└── reports/
    └── ...
└── README.md
└── ...
```

Here `train_config.py` is a copy of `train.py` in which you can change some default parameters that are going to be the same across all subexperiments you plan to do.

`drr_param_sweep.csv` is a file in which you specify the parameters you want to sweep over. Each row in this file is one run and each column must have the name of one of the keys of the `args` dictionary specified in `train.py`. For example,
```drr_param_sweep.csv
training_procedure,alpha,beta
drr_procedure,1e-5,5.0
drr_procedure,1e-7,10.0
```

`submit_experiment.sh` can then look as follows:
```submit_experiment.sh
# submit_experiment.sh

CONFIG="drr_param_sweep"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGPATH="$(realpath "$DIR/../experiment_scripts/${CONFIG}.csv")"
N=$(($(wc -l < $CONFIGPATH) - 1))  # subtract 1 for header

SBATCHPATH="$(realpath "$DIR/launch_experiment.sh")"

sbatch --array=0-$N --job-name=$CONFIG $SBATCHPATH
```
Executing this file (`./slurm_scripts/submit_experiment.sh`) launches one batch job for each row in `drr_param_sweep.csv` by calling (and passing the arguments in that row to) `launch_experiment.sh`, which, depending on the hardware you use, might look as follows:
```launch_experiment.sh
#!/bin/bash -l

# Standard output and error:
#SBATCH --output=./reports/%x_%j.out
#SBATCH --error=./reports/%x_%j.err
#SBATCH --open-mode=append

#SBATCH --constraint="gpu-bw"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000

#SBATCH --time=23:59:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.mail@somemail.com

#SBATCH --job-name=alpha_sweeps
#SBATCH --array=0-8

ulimit -n 65535 # increase the maximum number of open file descriptors

cd ~/path/to/efficient-compression/compressing_transformers/
mkdir -p reports

eval "$(conda shell.bash hook)"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wikipedia-experiments

# export environment variables to python script
export PYTHONUNBUFFERED=1
export MASTER_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=29500

echo "MASTER_IP: $MASTER_IP"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_PORT: $MASTER_PORT"
echo "[$SLURM_JOB_NAME]"

# run python script
srun --ntasks=$WORLD_SIZE --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    bash -c "export RANK=\$SLURM_PROCID; \
    echo \"RANK: \$RANK\"; \
    python -u ~/path/to/efficient-compression/compressing_transformers/experiment_scripts/train_config.py --config $SLURM_JOB_NAME --index $SLURM_ARRAY_TASK_ID"
```

**Important Note**: Make sure the total number of gpus multiplied with the batch size per gpu (specified in the transformer config) equals the desired batch size (for example 128 in our experiments).

Also note that the lines
```
#SBATCH --job-name=alpha_sweeps
#SBATCH --array=0-8
```
are just default values which are going to be overwritten by the commands in `submit_experiments.sh`. The job-name is overwritten by the name of the csv file (in our case `drr_param_sweep`) and the array number is overwritten by the number of rows in that csv file.

When the runs are done, you will see a subfolder in `experiment-results` with an info csv file in which all important parameters and metrics are logged and an artifact subfolder where models are stored for later use if desired.

Now if you want to do a second experiment, all you have to do is to create a file similar to `drr_param_sweep.csv` and change CONFIG variable in `submit_experiment.sh` to reflect the new file name and submit again. Everything else can remain unchanged.