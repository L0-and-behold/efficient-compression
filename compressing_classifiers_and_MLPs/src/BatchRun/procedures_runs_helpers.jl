"""
Helper functions shared by
- single_run_routine_teacherstudent.jl
- single_run_routine_classifier.jl
"""

########## Helper functions ##########
using DataFrames, Dates, CSV, Plots, CUDA
import Lux

"""
    do_small_run_to_trigger_precompilation(optimization_procedure, throwaway_tstate, train_set, validation_set, test_set, loss_fctn, args)

Run a small training procedure to trigger precompilation for more accurate timing measurements.
This function has side effects on the passed tstate.

# Arguments
- `optimization_procedure::Function`: The training procedure to precompile
- `throwaway_tstate::Lux.Training.TrainState`: Training state that will be modified during precompilation
- `train_set::Vector`: Training dataset (only first element will be used)
- `validation_set::AbstractArray`: Validation dataset
- `test_set::AbstractArray`: Test dataset
- `loss_fctn::Function`: Loss function
- `args`: Configuration parameters
"""
function do_small_run_to_trigger_precompilation(
    optimization_procedure::Function, 
    throwaway_tstate::Lux.Training.TrainState, 
    train_set::Vector, 
    validation_set::AbstractArray, 
    test_set::AbstractArray, 
    loss_fctn::Function, 
    args)

    println("Do small run to trigger precompilation of involved functions...")

    local_trainset = [train_set[1]]
    local_args = deepcopy(args)

    local_args.min_epochs = 1
    local_args.max_epochs = max(2, min(20, Int(round(args.max_epochs / 50))))
    local_args.finetuning_max_epochs = 1
    local_args.smoothing_window = 10
    local_args.binary_search_resolution = 0.01
    local_args.shrinking_from_deviation_of = 10.0

    try # errors during this run are not critical. we just want to trigger precompilation of the involved functions
        optimization_procedure(local_trainset, validation_set, test_set, throwaway_tstate, loss_fctn, local_args)
    catch e
        if isa(e, ArgumentError) && occursin("A weight matrix was completely deleted", e.msg)
            # println("Caught expected error during precompilation: ", e.msg)
            # println("This error is expected and doesn't affect the actual training.")
        else
            @warn "Unexpected error during precompilation: " exception=(e, catch_backtrace())
        end
    end

    println("Done with precompilation. Now starting the actual training.")
end

"""
    save_CSV(artifact_folder, data::Union{Vector{<:Tuple}, DeviceIterator}, title::String)

Save tuple data to a CSV file with epoch and data columns.

# Arguments
- `artifact_folder`: Directory to save the CSV file
- `data::Union{Vector{<:Tuple}, DeviceIterator}`: Vector of tuples containing epoch and value pairs
- `title::String`: Base name for the CSV file (without extension)
"""
function save_CSV(artifact_folder, data::Union{Vector{<:Tuple}, DeviceIterator}, title::String)
    epochs = [d[1] for d in data]
    values = [d[2] for d in data]
    df = DataFrame(epoch=epochs, data=values)
    filename = joinpath(artifact_folder, title * ".csv")
    CSV.write(filename, df)
end

"""
    save_CSV(artifact_folder, data::Vector{<:Number}, title::String)

Save numeric vector data to a CSV file with a single data column.

# Arguments
- `artifact_folder`: Directory to save the CSV file
- `data::Vector{<:Number}`: Vector of numeric values
- `title::String`: Base name for the CSV file (without extension)
"""
function save_CSV(artifact_folder, data::Vector{<:Number}, title::String)
    filename = joinpath(artifact_folder, title*".csv")
    CSV.write(filename, DataFrame(data=data))
end


"""
    do_and_save_plot(artifact_folder, logs::Dict, key_1::String, key_2::String; ft_starts_at=0)

Create and save a plot comparing two data series from logs.

# Arguments
- `artifact_folder`: Directory to save the plot
- `logs::Dict`: Dictionary containing the data to plot
- `key_1::String`: Key for the first data series
- `key_2::String`: Key for the second data series
- `ft_starts_at=0`: Optional epoch to mark as the start of fine-tuning (0 means no marking)
"""
function do_and_save_plot(artifact_folder, logs::Dict, key_1::String, key_2::String; ft_starts_at=0)
    if ft_starts_at == 0
        p = do_plot(logs, key_1, key_2)
    else
        p = do_plot(logs, key_1, key_2, ft_starts_at)
    end
    savefig(p, joinpath(artifact_folder, key_1 * "_and_" * key_2 * ".pdf"))
end

"""
    do_and_save_plot(artifact_folder, logs::Dict, title::String; ft_starts_at=0)

Create and save a plot of a single data series from logs.

# Arguments
- `artifact_folder`: Directory to save the plot
- `logs::Dict`: Dictionary containing the data to plot
- `title::String`: Key for the data series to plot
- `ft_starts_at=0`: Optional epoch to mark as the start of fine-tuning (0 means no marking)
"""
function do_and_save_plot(artifact_folder, logs::Dict, title::String; ft_starts_at=0)
    if ft_starts_at == 0
        p = do_plot(logs, title)
    else
        p = do_plot(logs, title, ft_starts_at)
    end
    savefig(p, joinpath(artifact_folder, title * ".pdf"))
end

"""
    do_plot(logs::Dict, title::String) -> Plots.Plot

Create a plot of a single data series from logs.

# Arguments
- `logs::Dict`: Dictionary containing the data to plot
- `title::String`: Key for the data series to plot

# Returns
- `Plots.Plot`: The generated plot
"""
function do_plot(logs::Dict, title::String)
    X = logs["epochs"]
    Y = logs[title]
    xlabel = "Epochs"
    ylabel = title
    p = plot(X, Y, label=title, xlabel=xlabel, ylabel=ylabel)
return p
end

"""
    do_plot(logs::Dict, title::String, ft_start_at::Int) -> Plots.Plot

Create a plot of a single data series from logs with a vertical line marking fine-tuning start.

# Arguments
- `logs::Dict`: Dictionary containing the data to plot
- `title::String`: Key for the data series to plot
- `ft_start_at::Int`: Epoch to mark as the start of fine-tuning

# Returns
- `Plots.Plot`: The generated plot
"""
function do_plot(logs::Dict, title::String, ft_start_at::Int)
    p = do_plot(logs, title)
    vline!([ft_start_at], label="fine tuning", linestyle=:dash, color=:red)
    return p
end

"""
    do_plot(logs::Dict, key_1::String, key_2::String) -> Plots.Plot

Create a plot comparing two data series from logs.

# Arguments
- `logs::Dict`: Dictionary containing the data to plot
- `key_1::String`: Key for the first data series
- `key_2::String`: Key for the second data series

# Returns
- `Plots.Plot`: The generated plot
"""
function do_plot(logs::Dict, key_1::String, key_2::String)
    X = logs["epochs"]
    Y1 = logs[key_1]
    Y2 = logs[key_2]
    title = key_1 * " and " * key_2
    p = plot(X, Y1, xlabel="Epochs", label = key_1, ylabel=title)
    plot!(X, Y2, label=key_2)
    return p
end

"""
    do_plot(logs::Dict, key_1::String, key_2::String, ft_starts_at::Int) -> Plots.Plot

Create a plot comparing two data series from logs with a vertical line marking fine-tuning start.

# Arguments
- `logs::Dict`: Dictionary containing the data to plot
- `key_1::String`: Key for the first data series
- `key_2::String`: Key for the second data series
- `ft_starts_at::Int`: Epoch to mark as the start of fine-tuning

# Returns
- `Plots.Plot`: The generated plot
"""
function do_plot(logs::Dict, key_1::String, key_2::String, ft_starts_at::Int)
    p = do_plot(logs, key_1, key_2)
    vline!([ft_starts_at], label="fine tuning", linestyle=:dash, color=:red)
    return p
end

"""
    do_and_save_plots(artifact_folder, logs, args)

Create and save multiple plots for training visualization.

# Arguments
- `artifact_folder`: Directory to save the plots
- `logs`: Dictionary containing all log data
- `args`: Configuration parameters containing settings for plot generation
"""
function do_and_save_plots(artifact_folder, logs, args)
    do_and_save_plot(artifact_folder, logs, "train_loss", "val_loss", ft_starts_at=logs["converged_at"][1])
    do_and_save_plot(artifact_folder, logs, "epoch_execution_time", ft_starts_at=logs["converged_at"][1])

    if OptimizationProcedures.is_saturated(logs["val_loss"], args.smoothing_window; min_possible_deviation=args.shrinking_from_deviation_of)
        p = OptimizationProcedures.is_saturated(logs["val_loss"], args.smoothing_window; min_possible_deviation=args.shrinking_from_deviation_of, plot_graph=true, return_plot=true)
        savefig(p, joinpath(artifact_folder, "saturated.pdf"))
    end
end