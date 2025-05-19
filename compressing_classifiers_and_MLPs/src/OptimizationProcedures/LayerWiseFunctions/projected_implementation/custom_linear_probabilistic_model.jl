

using Lux, Random, WeightInitializers, Revise # Importing `Lux` also gives you access to `LuxCore`

"""
    This struct serves as a linear neural network layer in julia's Lux framework that incorporates additional weights that encode the probabilities (hence Pro-linear) of a mask in the context of the Probabilistic Exact Gradient Pruning (PEGP) method. It is designed to work with the projected gradient descent implementation.
"""
struct Prolinear <: LuxCore.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    initial_w_params
    initial_pw_params
    initial_b_params
    initial_pb_params
end

function Prolinear(in_dims::Int, out_dims::Int; initial_w_params=glorot_uniform, initial_pw_params=Lux.zeros32, initial_b_params=glorot_uniform, initial_pb_params=Lux.zeros32)
    return Prolinear(in_dims, out_dims, initial_w_params, initial_pw_params, initial_b_params, initial_pb_params)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::Prolinear)
    return (w=l.initial_w_params(rng, l.out_dims, l.in_dims), pw=l.initial_pw_params(rng, l.out_dims, l.in_dims), b=l.initial_b_params(rng, l.out_dims), pb=l.initial_pb_params(rng, l.out_dims))
end

LuxCore.initialstates(::AbstractRNG, ::Prolinear) = NamedTuple()
LuxCore.parameterlength(l::Prolinear) = l.out_dims * l.in_dims + l.out_dims
LuxCore.statelength(::Prolinear) = 0

function (l::Prolinear)(x::AbstractMatrix, ps, st::NamedTuple)
    y = (ps.w .* ps.pw) * x .+ (ps.b .* ps.pb)
    return y, st
end


## Example

# l = Prolinear(2,3)

# rng = Random.default_rng()
# Random.seed!(rng, 0)

# ps, st = LuxCore.setup(rng, l)

# println("Parameter Length: ", LuxCore.parameterlength(l), "; State Length: ",
#     LuxCore.statelength(l))

# x = randn(rng, Float32, 2, 5)

# LuxCore.apply(l, x, ps, st) # or `l(x, ps, st)`
