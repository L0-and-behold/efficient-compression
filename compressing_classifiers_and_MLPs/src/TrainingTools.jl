module TrainingTools

using BSON
import Lux
using Random


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
    BSON.@save filename train_state model rng
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
    BSON.@load filename train_state model rng
    return train_state, model, rng
end



end