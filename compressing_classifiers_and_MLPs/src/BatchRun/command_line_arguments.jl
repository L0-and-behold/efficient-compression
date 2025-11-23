"""
    get_sub_batch(experiment_name::String, batch::Vector{T}) where T <: Tuple

Splits a batch of runs into sub-batches and returns the portion corresponding to the specified sub-batch index.

# Arguments
- `experiment_name::String`: The base name of the experiment. If the batch is split, the sub-batch identifier is appended.
- `batch::Vector{T}`: A vector of tuples representing the full batch of runs.

# Returns
- `experiment_name::String`: Updated experiment name, potentially including sub-batch info.
- `batch::Vector{T}`: The sub-batch of runs corresponding to the specified sub-batch index.

This function uses command line arguments `--num_sub_batches` and `--sub_batch` to determine which part of the batch to execute.
"""
function get_sub_batch(experiment_name::String, batch::Vector{T}) where T <: Tuple
    parsed_args = parse_batch_distribution_args()
    num_sub_batches = parsed_args["num_sub_batches"]
    sub_batch = parsed_args["sub_batch"]
    batch_size = ceil(Int, length(batch) / num_sub_batches)
    start_index = (sub_batch - 1) * batch_size + 1
    end_index = min(sub_batch * batch_size, length(batch))
    batch = batch[start_index:end_index]
    
    if num_sub_batches > 1
        experiment_name *= "_$sub_batch-$num_sub_batches"
        println("Running sub-batch $sub_batch of $num_sub_batches")
    end

    return experiment_name, batch
end

"""
    parse_batch_distribution_args()

Parses command line arguments related to sub-batch distribution.

# Command Line Arguments
- `--num_sub_batches::Int`: Total number of sub-batches (default = 1).
- `--sub_batch::Int`: Index (1-based) of the sub-batch to run (default = 1).

# Returns
- `Dict{String, Any}`: A dictionary with parsed values for `num_sub_batches` and `sub_batch`.
"""
function parse_batch_distribution_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--num_sub_batches"
            help = "Total number of sub-batches"
            arg_type = Int
            default = 1
        "--sub_batch"
            help = "Current sub-batch to run (1-indexed)"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end
