module TrainingTools

    using BSON, Random
    import Lux
    using CUDA

    export save_train_state, load_train_state, to_cpu, to_gpu

    """
        to_cpu(x::Any)

        to_cpu(x) transfers all elements in x recursively to cpu.

        This is necessary because a simple `train_state |> cpu_device()` can not properly handle the recursive structures in the train_state struct.
    """
    to_cpu(x::CuArray) = Array(x)
    to_cpu(x::AbstractArray) = x
    to_cpu(x::Union{Number,Symbol,String,Nothing,Function,DataType,Type}) = x
    function to_cpu(x::T) where {T}
        isstructtype(T) || return x
        fields = fieldnames(T)
        isempty(fields) && return x

        new_fields = map(f -> to_cpu(getfield(x, f)), fields)

        # Strip type parameters, let constructor re-infer them from new_fields
        constructor = T.name.wrapper
        try
            return constructor(new_fields...)
        catch
            return x  # give up gracefully rather than erroring the whole tree
        end
    end
    to_cpu(x::NamedTuple) = NamedTuple{keys(x)}(map(to_cpu, values(x)))
    to_cpu(x::Tuple) = map(to_cpu, x)

    """
        to_gpu(x::Any)

        to_gpu(x) transfers all Float Arrays in x recursively to gpu.

        This function can be used to transfer the parameters and optimizer states of a loaded model back to the gpu.
    """
    to_gpu(x::CuArray) = x  # already on GPU
    to_gpu(x::AbstractArray{<:AbstractFloat}) = CuArray(x)  # only float arrays -> GPU
    to_gpu(x::AbstractArray) = x  # e.g. arrays of Bool/Int indices are kept on CPU
    to_gpu(x::Union{Number,Symbol,String,Nothing,Function,DataType,Type}) = x
    function to_gpu(x::T) where {T}
        isstructtype(T) || return x
        fields = fieldnames(T)
        isempty(fields) && return x

        new_fields = map(f -> to_gpu(getfield(x, f)), fields)
        constructor = T.name.wrapper
        try
            return constructor(new_fields...)
        catch
            return x
        end
    end
    to_gpu(x::NamedTuple) = NamedTuple{keys(x)}(map(to_gpu, values(x)))
    to_gpu(x::Tuple) = map(to_gpu, x)

    """
        save_train_state(train_state::Lux.Training.TrainState, model, rng, filename)

    Save a Lux training state, model, and random number generator to a BSON file.

    # Arguments
    - `train_state::Lux.Training.TrainState`: The training state to save
    - `model`: The Lux model to save
    - `rng`: The random number generator state to save
    - `filename`: Path where the BSON file will be saved

    # Example
    ```julia
    save_train_state(tstate, model, Random.GLOBAL_RNG, "train_state.bson")
    ```
    """
    function save_train_state(train_state::Lux.Training.TrainState, model, rng, filename) 
        train_state_cpu = to_cpu(train_state)
        BSON.@save filename train_state_cpu model rng
        println("TrainState and model saved to $filename")
    end

    """
        load_train_state(filename) -> (train_state, model, rng)

    Load a previously saved training state, model, and random number generator from a BSON file.

    # Arguments
    - `filename`: Path to the BSON file to load

    # Returns
    - `train_state`: The loaded Lux training state
    - `model`: The loaded Lux model
    - `rng`: The loaded random number generator state

    # Example
    ```julia
    tstate, model, rng = load_train_state("train_state.bson")
    ```
    """
    function load_train_state(filename)
        BSON.@load filename train_state_cpu model rng
        return train_state_cpu, model, rng
    end

    # """
    #     value_equal can serve to compare to_gpu(to_cpu(train_state)) with train_state.
    #     It returns value_equal(train_state, to_gpu(to_cpu(train_state))) == true
    # """
    # function value_equal(a::T, b::T) where T
    #     if isstructtype(T) && !isempty(fieldnames(T))
    #         return all(value_equal(getfield(a,f), getfield(b,f)) for f in fieldnames(T))
    #     else
    #         return a == b
    #     end
    # end
    # value_equal(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(Array(a) .== Array(b))
    # value_equal(a, b) = a == b
end # module TrainingTools