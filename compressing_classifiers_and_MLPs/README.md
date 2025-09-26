# Compressing Classifiers and MLPs

Codebase for the experiments from our paper regarding the classifier and teacherstudent experiments.

## Table of Contents
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Experiment Parameters and Setup](#experiment-parameters-and-setup)
  - [General Parameters](#general-parameters)
  - [Dataset and Training Parameters](#dataset-and-training-parameters)
- [Standard Dataset Configurations](#standard-dataset-configurations)
  - [Cifar with VGG](#cifar-with-vgg)
  - [Mnist with Lenet 5 Caffe](#mnist-with-lenet-5-caffe)
  - [Mnist with Lenet 300100](#mnist-with-lenet-300100)
  - [Teacherstudent](#teacherstudent)
- [Parallelized Execution with Subbatches and SLURM](#parallelized-execution-with-subbatches-and-slurm)

## Quick Start

The code assumes, that a CUDA gpu-device is present for the classifier experiments. The teacherstudent experiments run well on a cpu (faster even).

Make sure that you have Julia installed https://docs.julialang.org/en/v1/manual/installation/

Make sure that you are operating from the correct path and start julia
```shell
cd path/to/project/efficient-compression/compressing_classifiers_and_MLPs
julia
```

In julia in shell, the environment can be instantiated like this 
```julia
using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
```

Now, an experiment can be run
```shell
julia run_an_experiment.jl
```
where the default settings in the file `run_an_experiment.jl` serve as a simple example.

The main file `run_an_experiment.jl` also contains doc-strings which serve as a walkthrough on how to set up and run an experiment.

## Experiment Parameters and Setup

After the experiment finishes, the results are saved into a folder including a runs.csv file containing parameters and calculated metrics as well as an artifact folder with the trained model parameters and loss curves.

Set the path for results storage in `run_an_experiment.jl` by changing:

`path_to_db = joinpath(pwd(), "experiment-results")`
and
`experiment_name = "example-experiment"`

The other training settings are controlled by the `args=TrainArgs()` object. In the following we document what each field of `args` does.

### General Parameters

| Parameter| Description|
| -- | -- |
| `architecture`| Specifies the model architecture. Can be a callable (e.g., `VGG`, `Lenet_5`) from the `OptimizationProcedures` module for classification tasks. |
| `optimization_procedure`| Defines the training pipeline, selected from procedures in `OptimizationProcedures`, such as `PMMP_procedure`, `RL1_procedure`, `DRR_procedure`, or `layerwise_procedure`. |
| `dataset` | For classification tasks, selects the dataset constructor from `DatasetsModels`, such as `MNIST_data` or `CIFAR_data`. No function in teacher-student settings.|
| `architecture_teacher` | For teacher-student experiments, specifies the teacher MLP architecture as a list of layer dimensions: `[input_dim, hidden_dims..., output_dim]`. |
| `architecture_student` | Analogous to `architecture_teacher`, defines the student MLP architecture. |
| `dtype` | Numerical precision used for model parameters and computations. Supported values: `Float32` (recommended) or `Float64`.|
| `dev` | Computation device: use `Lux.cpu_device()` for CPU or `Lux.gpu_device()` for GPU. CPU is recommended for MLPs; GPU for convolutional architectures. |
| `verbose` | Enables more detailed logging during training if set to `true`.|
| `optimizer` | Optimizer instance from the `Optimisers` package used for weight updates.|
| `lr`| Initial learning rate used by the optimizer. |
| `min_epochs`| Minimum number of training epochs before early stopping or pruning is allowed. |
| `max_epochs`| Maximum number of training epochs. |
| `α` | Weight of the regularization term in applicable procedures. |
| `β` | Sharpness parameter for the DRR procedure. |
| `ρ` | Coefficient for L2 weight regularization in DRR and RL1 procedures. |
| `L1_alpha`| Coefficient for additional L1 regularization term in PMMP procedure. It adds L1_alpha times the L1-norm of the parameters of the unregularized objective to the PMMP loss function. |
| `tolerated_relative_loss_increase` | Pruning threshold parameter (`δ`) for TAMADE: finds largest pruning threshold such that post-pruning loss ≤ (1 + δ) × pre-pruning loss. |
| `NORM`| If `true`, applies layerwise normalization to `α` and `ρ` in DRR. |
| `layer_NORM`| If set to `true`, the `NORM` normalization described above divides by the number of neurons in a given layer to normalize. If set to `false`, and the layer consists of convolutional blocks (in a CNN), then it divides by the number of neurons in a convolution block to normalize. |

---

### Dataset and Training Parameters

| Parameter | Description|
| --- | --- |
| `train_set_size` | Number of training samples used.|
| `train_batch_size` | Number of samples per training batch. |
| `val_set_size`| Number of validation samples used.|
| `val_batch_size` | Number of samples per validation batch. |
| `test_set_size`| Number of test samples used.|
| `test_batch_size`| Number of samples per test batch. |
| `noise` | For MLPs, specifies the variance of Gaussian noise added to the teacher's output during synthetic data generation. |
| `gauss_loss`| If `true`, uses a Gaussian loss ; otherwise, uses mean squared error (MSE). Applies only to MLPs (teacherstudent). |
| `random_gradient_pruning`| If set to `true`, then Random Gradient Pruning is performed before and after pruning with TAMADE. |
| `shrinking` | If set to `true`, then pruning (including Random Gradient Pruning and TAMADE) is performed every `prune-window` epochs during training if the loss curve is saturated up to `shrinking_from_deviation_of` (and only if epoch number exceeds `min_epochs`). |
| `delete_neurons` | If set to `true`, then neurons without remaining connections are deleted for MLP architectures after each pruning step. |
| `binary_search_resolution` | Resolution parameter for the binary search used in TAMADE pruning (i.e., how finely the optimal pruning threshold is determined before breaking the algorithm).|
| `shrinking_from_deviation_of`| If `shrinking` is set to `true`, then pruning (including Random Gradient Pruning and TAMADE) is performed every `prune-window` epochs during training if the loss curve is saturated up to `shrinking_from_deviation_of` (and only if epoch number exceeds `min_epochs`). Thus, the higher `shrinking_from_deviation_of`, the less saturated the loss has to be before pruning begins. |
| `prune_window`| Frequency (in epochs) at which pruning is applied, starting after `min_epochs`. |
| `smoothing_window` | Size of the moving window used to assess training saturation based on loss stability. |
| `finetuning_shrinking` | Like `shrinking` explained above but during the finetuning phase. |
| `finetuning_min_epochs`| Minimum number of epochs for fine-tuning after pruning.|
| `finetuning_max_epochs`| Maximum number of epochs for fine-tuning. |
| `layerwise_pruning`| If set to `true`, then layerwise_pruning is performed every `prune_window` epochs during training (in addition to Random Gradient Pruning and TAMADE) once the number of epochs has exceeded `min_epochs` and once the loss is saturated up to `shrinking_from_deviation_of`. |
| `finetuning_layerwise_pruning` | Like `layerwise_pruning` explained above but during finetuning phase. |
| `layerwise_pruning_alpha`| Regularization coefficient for layerwise pruning optimization. |
| `layerwise_pruning_lr` | Learning rate used in layerwise pruning optimization.|
| `layerwise_pruning_mask_start_value` | Initial value for pruning masks in layerwise optimization. |
| `log_val_loss`| If `true`, logs validation loss during training and saves as an artifact after training. |
| `converge_val_loss`| If `true`, convergence is determined based on validation loss; if `false`, based on training loss. |
| `finetuning_converge_val_loss` | Like `converge_val_loss` explained above but during finetuning phase. |
| `logs`| -a placeholder for a variable which the procedures write to-|
| `multiply_mask_after_each_batch` | If `true`, applies pruning masks after every training batch; otherwise, only at designated pruning stages. |
| `initial_p_value`| Initial value of the `p` parameter for PMMP optimization.|
| `initial_u_value`| Initial value of the `u` parameter for PMMP optimization.|
| `u_value_multiply_factor`| Scaling factor applied to `u` during PMMP optimization.|
| `seed`| Sets the random seed for reproducibility across training runs. |
## Standard Dataset Configurations

We provide standard configurations for our experiments.
The other arguments are as the defaults in `src/TrainArgs.jl` or are variables.

### Cifar with VGG

```julia
single_run_routine = single_run_routine_classifier

args.architecture = VGG
args.dataset = CIFAR_data
args.train_batch_size = 500
args.smoothing_window = 20
args.min_epochs = 30
args.max_epochs = 300
args.finetuning_min_epochs = 10
args.finetuning_max_epochs = 50
args.train_set_size = "see dataset"
args.val_set_size = "see dataset"
args.val_batch_size = "val_set_size"
args.test_set_size = "see dataset"
args.test_batch_size = "test_set_size"
args.noise = 0f0
args.prune_window = 10
args.shrinking_from_deviation_of = 1e-2
args.gauss_loss = false
args.dev = gpu_device()
```

### Mnist with Lenet 5 Caffe

```julia
single_run_routine = single_run_routine_classifier

args.architecture = Lenet_5_Caffe
args.dataset = MNIST_data
args.train_batch_size = 500
args.max_epochs = 5000
args.finetuning_max_epochs = 1000
args.min_epochs = 300
args.train_set_size = "see dataset"
args.val_set_size = "see dataset"
args.val_batch_size = "val_set_size"
args.test_set_size = "see dataset"
args.test_batch_size = "test_set_size"
args.noise = 0f0
args.prune_window = 10
args.gauss_loss = false
args.dev = gpu_device()
```

### Mnist with Lenet 300100

```julia
single_run_routine = single_run_routine_classifier

args.architecture = Lenet_MLP
args.dataset = MNIST_data
args.train_batch_size = 500
args.max_epochs = 5000
args.finetuning_max_epochs = 1000
args.min_epochs = 300
args.train_set_size = "see dataset"
args.val_set_size = "see dataset"
args.val_batch_size = "val_set_size"
args.test_set_size = "see dataset"
args.test_batch_size = "test_set_size"
args.noise = 0f0
args.prune_window = 10
args.shrinking = false
args.dev = gpu_device()
args.gauss_loss = false
```

### Teacherstudent

```julia
single_run_routine = single_run_routine_teacherstudent

args.architecture_teacher = [2, 5, 8, 1]
args.architecture_student = [2, 25, 25, 1]
args.max_epochs = 50000
args.min_epochs = 5000
args.prune_window = 50
args.finetuning_max_epochs = 10000
args.val_set_size = 1000
args.val_set_size = args.val_set_size
args.test_set_size = 5000
args.test_batch_size = args.test_set_size
args.shrinking = false
args.smoothing_window = 500
args.dev = cpu_device()
args.dataset = "teacher_student"
args.architecture = "teacher_student"
args.delete_neurons = false # turn off for debug
args.lr = 5f-4
```

## Parallelized Execution with Subbatches and SLURM

To run an experiment with several workers, we recommend the following construction.

1. Set up a simple executable which starts all jobs, specifying the job name as well as how many subbatches are used.

```sh
#!/bin/bash

# what slurm script to use
SLURM_SCRIPT=~/path/to/parse_job.sh

# how many sub-batches to split the experiment into
experiment_name="example_name"
number_of_sub_batches=1


for batch_idx in $(seq 1 $number_of_sub_batches); do
echo "Submitting job for sub-batch ${batch_idx}/${number_of_sub_batches}"
job_name="${experiment_name}_${batch_idx}"
sbatch_output=$(sbatch --job-name="$job_name" "$SLURM_SCRIPT" "$number_of_sub_batches" "$batch_idx")
sbatch_exit_code=$?
if [ $sbatch_exit_code -eq 0 ]; then
echo "Job submitted successfully: $sbatch_output"
else
echo "Error submitting job: $sbatch_output"
fi
sleep 0.1
done

echo "All jobs submitted!"
```

2. Set up a slurm job script, which accepts arguments passed from the abovementioned executable

```slurm
#!/bin/bash -l

#SBATCH --output=./reports/%x_%j.out
#SBATCH --error=./reports/%x_%j.err
#SBATCH --open-mode=append

#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu"

export JULIA_STUDIO_UNBUFFERED=1

cd ~/path/to/project

mkdir -p reports

######
# Launch this script in the `launch_subexperiments` wrapper
######

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

srun --ntasks=$WORLD_SIZE --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
bash -c "export RANK=\$SLURM_PROCID; \
echo \"RANK: \$RANK\"; \
julia --threads auto ~/path/to/run_an_experiment.jl --num_sub_batches \"$1\" --sub_batch \"$2\"" 2>&1
```

These two files together allow to run an experiment with many workers in parallel.

The experiment results in the runs.csv files of the subexperiments can easily be merged, eg. by using an executable similar to:

```sh
#!/bin/bash

# Usage: ./merge_experiments.sh experiment_name number_subexperiments

set -e

EXPERIMENT_NAME=$1
NUM_SUBEXP=$2
MERGED_DIR="${EXPERIMENT_NAME}"
MERGED_CSV="${MERGED_DIR}/runs.csv"

echo "Step 1: Merging runs.csv files..."
> "$MERGED_CSV"# clear or create the merged csv file

for i in $(seq 1 "$NUM_SUBEXP"); do
SUBDIR="${EXPERIMENT_NAME}_${i}-${NUM_SUBEXP}"
CSV="${SUBDIR}/runs.csv"

if [ "$i" -eq 1 ]; then
cat "$CSV" >> "$MERGED_CSV"
else
tail -n +2 "$CSV" >> "$MERGED_CSV"
fi
done

echo "done."
```

We provide these scripts in the readme rather than providing executables, as their exact implementation highly depends on the exact machine worked on.

It is also possible to just start the `run_an_experiment.jl` script several times in parallel without using sub-experiments. This approach might result in race conditions, however, if too many workers are working on the same directory in parallel.