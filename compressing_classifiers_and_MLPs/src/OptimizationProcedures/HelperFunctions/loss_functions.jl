using UnPack, Lux
using OneHotArrays, Revise
using Accessors

# The following file defines the loss functions for the different optimization and compression procedures (like DRR, PMMP or RL1).
# Each loss has an associated struct that determines the optimization steps during training via multiple dispatch.
# New losses and procedures can thus be defined by creating a new loss struct in this file and by adding a corresponding update_state! function in update_functions.jl which takes this struct as input. 

abstract type LossFunction end

const logitcrossentropy = Lux.CrossEntropyLoss(; logits=Val(true))
# Test:
# y = [1  0  0  0  1
#      0  1  0  1  0
#      0  0  1  0  0]
# model_prediction = reshape(-7:7, 3, 5) .* 1f0
# CEL1 = CrossEntropyLoss()(softmax(model_prediction), y)
# CEL2 = logitcrossentropy(model_prediction, y)
# CEL1 ≈ CEL2 # true

# function sse_loss(model, ps::NamedTuple, st::NamedTuple, (X,y)::Tuple) # use Lux.MSELoss() instead and rescale alpha
#     yp, st = model(X,ps,st)
#     innerVec = (y .- yp).^2
#     loss = sum(innerVec)
#     stats = nothing
#     return loss, st, stats
# end

# function mse_loss(model, ps::NamedTuple, st::NamedTuple, (X,y)::Tuple) # use Lux.MSELoss() instead
#     yp, st = model(X,ps,st)
#     innerVec = (y .- yp).^2
#     loss = sum(innerVec)/length(innerVec)
#     stats = nothing
#     return loss, st, stats
# end

#################

function recursively_multiply!(A, B) # this can not be used inside of loss functions to modify parameters because of array mutation (in place operations). Use recursive_map below instead.
    for (a,b) in zip(A, B)
        if isa(a, NamedTuple) && !isempty(a) # && isa(b, NamedTuple)
            recursively_multiply!(a, b)
        elseif isa(a, AbstractArray) # && isa(b, AbstractArray{T} where T)
            a .*= b
        end
    end
end

function recursive_map(f, nt1::NamedTuple, nt2::NamedTuple)
    return NamedTuple{keys(nt1)}(map( (v1, v2) -> 
        isa(v1, NamedTuple) ? recursive_map(f, v1, v2) :
        isa(v1, AbstractArray) ? f(v1, v2) : v1, values(nt1), values(nt2) ))
end
multiply_mask(A,B) = A .* B

#################

function accuracy(tstate, dataset)::Float32
    if haskey(tstate.parameters, :p)
        return accuracy(tstate.model, tstate.parameters.p, tstate.states.st, dataset)
    end
    return accuracy(tstate.model, tstate.parameters, tstate.states, dataset)
end

function accuracy(model, ps, st, dataset)::Float32
    total_correct, total = 0, 0
    stt = Lux.testmode(st)
    for (x, y) in dataset
        target_class = onecold(y)
        predicted_class = onecold(first(model(x, ps, stt)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end



function masked_loss(loss_params, model, ps, st, batch)
    if haskey(st, :mask)
        loss, stst, stats = loss_params.loss_f(model, recursive_map(multiply_mask, ps.p, st.mask), st.st, batch)
        @reset st.st = stst
        return loss, st, stats
    else
        return loss_params.loss_f(model, ps.p, st, batch)
    end
end

function Gauss_loss(loss_params, model, ps::NamedTuple, st::NamedTuple, batch)
    sigma = sum(ps.sigma) # sigma has only 1 element but sum prevents scalar indexing. Nevertheless, it has to remain on GPU for consistency with other params in tstate.
    

    if haskey(st, :mask)
        unaugmented_loss, stst, stats = loss_params.loss_f(model, recursive_map(multiply_mask, ps.p, st.mask), st.st, batch)
        @reset st.st = stst
    else
        unaugmented_loss, st, stats = loss_params.loss_f(model, ps.p, st, batch)
    end

    # small_const = 1f-12
    # division_const = 20f0
    # if sigma^2 < unaugmented_loss/division_const + small_const
    #     loss = unaugmented_loss / (1 * (sigma^2+unaugmented_loss/division_const+small_const) * log(2)) + 0.5 * log2(0.5 * (sigma^2+unaugmented_loss/division_const+small_const)) # adding unaugmented_loss/division_const to sigma stabilizes the loss. It means that sigma -> 0 can not increase unaugmented_loss by more than division_const times. Since sigma -> 0 should only happen when unaugmented_loss is also very close to 0, this stabilizes without causing significant distortion of the loss. Furthermore, it works regardless of the size of unaugmented_loss.
    # else
        loss = unaugmented_loss / (2f0 * sigma^2 * log(2f0)) + 0.5f0 * log2(sigma^2)
    # end

    return loss, st, stats
end

function DRR_NORM_modification(loss_fun, grad, weight, layer)
    scaling_factor = loss_fun.model_param_number / Lux.parameterlength(layer) / loss_fun.layernumber_model
    return scaling_factor .* DRR_modification(loss_fun, grad, weight, layer)
end
function DRR_NORM_layer_modification(loss_fun, grad, weight, layer)
    scaling_factor = loss_fun.model_param_number / loss_fun.layer_param_number[1] / loss_fun.layernumber_model
    return scaling_factor .* DRR_modification(loss_fun, grad, weight, layer)
end
function DRR_modification(loss_fun, grad, weight, layer)
    return loss_fun.alpha .* sign.(weight) .* loss_fun.beta .* exp.(- loss_fun.beta .* abs.(weight))
end

function L1_modification(loss_fun, grad, weight, layer)
    return loss_fun.alpha .* sign.(weight)
end
function L1_PMMP_modification(loss_fun, grad, weight, layer)
    return loss_fun.L1_alpha .* sign.(weight)
end

function L2_modification(loss_fun, grad, weight, layer)
    return loss_fun.rho .* 2 .* weight
end

function PMMP_gp_modification(sub_gp, sub_grads, sub_p, sub_pw, sub_pp, u, loss_fun)
    if hasproperty(loss_fun, :u_value_multiply_factor)
        mf = loss_fun.u_value_multiply_factor
    else
        mf = 1f0
    end
    sub_gp .= sub_grads .+ mf .* u .* 2 .* (sub_p .- sub_pw .* sub_pp)
end
function PMMP_gpw_modification(sub_gpw, sub_p, sub_pw, sub_pp, u, loss_fun)
    if hasproperty(loss_fun, :u_value_multiply_factor)
        mf = loss_fun.u_value_multiply_factor
    else
        mf = 1f0
    end
    sub_gpw .= mf .* (-1) .* u .* 2 .* (sub_p .- sub_pw .* sub_pp) .* sub_pp .+ mf .* u .* 2 .* sub_pw .* sub_pp .* (1 .- sub_pp)
end
function PMMP_gpp_modification(sub_gpp, sub_p, sub_pw, sub_pp, u, loss_fun)
    if hasproperty(loss_fun, :u_value_multiply_factor)
        mf = loss_fun.u_value_multiply_factor
    else
        mf = 1f0
    end
    sub_gpp .= mf .* (-1) .* u .* 2 .* (sub_p .- sub_pw .* sub_pp) .* sub_pw .+ mf .* u .* sub_pw.^2 .* (1 .- 2 .* sub_pp) .+ loss_fun.alpha
end
function PMMP_u_modification(sub_gu, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
    if hasproperty(loss_fun, :u_value_multiply_factor)
        mf = loss_fun.u_value_multiply_factor
    else
        mf = 1f0
    end
    sub_gu .= - mf .* (sub_p .- sub_pw .* sub_pp).^2 .- mf .* sub_pw.^2 .* sub_pp .* (1 .- sub_pp)
end


################################ structs #####################################


struct RL1_loss{R <: Number, L1 <: Function, L2 <: Function} <: LossFunction
    alpha::R
    rho::R
    fun1::Function
    fun2::Function
    loss_f::L2

    function RL1_loss(; alpha::R = 0.1f0, rho::R = 0f0, loss_f::L2 = Lux.MSELoss()) where {R,L2}
        return new{R, typeof(L1_modification), L2}(alpha, rho, L1_modification, L2_modification, loss_f)
    end
end
function (loss_params::RL1_loss)(model, ps::NamedTuple, st::NamedTuple, batch)
    return masked_loss(loss_params, model, ps, st, batch)
end


struct DRR{R <: Number, L2 <: Function} <: LossFunction
    alpha::R
    beta::R
    rho::R
    fun1::Function
    fun2::Function
    loss_f::L2
    model_param_number::R
    layernumber_model::R
    layer_param_number::Vector{R}

    function DRR(model_param_number, layernumber_model; alpha::R=1.0f0, beta::R=10f0, rho::R=0f0, loss_f::L2 = Lux.MSELoss(), fun1 = DRR_modification) where {R, L2}
        return new{R, L2}(alpha, beta, rho, fun1, L2_modification, loss_f, model_param_number, layernumber_model, [R(0)])
    end
end
function (loss_params::DRR)(model, ps::NamedTuple, st::NamedTuple, batch)
    return masked_loss(loss_params, model, ps, st, batch)
end

mutable struct PMMP{R, L <: Function} <: LossFunction
    grad_template
    model_param_number::R
    alpha::R
    rho::R
    u_value_multiply_factor::R
    loss_f::L
    fun_p
    fun_pw
    fun_pp
    fun_u
    fun_L2
    L1_alpha::R
    fun1

    function PMMP(grad_template, model_param_number; alpha::R=0.1f0, rho::R=0.0f0, u_value_multiply_factor::R=1f0, loss_f::L = Lux.MSELoss(), L1_alpha::R=0f0) where {R, L}
        return new{R, L}(grad_template, model_param_number, alpha, rho, u_value_multiply_factor, loss_f, PMMP_gp_modification, PMMP_gpw_modification, PMMP_gpp_modification, PMMP_u_modification, L2_modification, L1_alpha, L1_PMMP_modification)
    end
end
function (loss_params::PMMP)(model, ps::NamedTuple, st::NamedTuple, batch)
    return masked_loss(loss_params, model, ps, st, batch)
end

####### GAUSS #######

struct RL1_Gauss{R <: Number, L <: Function} <: LossFunction
    alpha::R
    rho::R
    loss_f::L

    function RL1_Gauss(; alpha::R = 0.1f0, rho::R = 0f0, loss_f::L = Lux.MSELoss()) where {R,L}
        return new{R,L}(alpha, rho, loss_f)
    end
end
function (loss_params::RL1_Gauss)(model, ps::NamedTuple, st::NamedTuple, batch)
    return Gauss_loss(loss_params, model, ps, st, batch)
end


struct DRR_Gauss{R, L <: Function} <: LossFunction
    NORM::Bool
    alpha::R
    beta::R
    rho::R
    loss_f::L

    function DRR_Gauss(; NORM=false, alpha::R=1.0f0, beta::R=10f0, rho::R=0f0, loss_f::L = Lux.MSELoss()) where {R, L}
        return new{R, L}(NORM, alpha, beta, rho, loss_f)
    end
end
function (loss_params::DRR_Gauss)(model, ps::NamedTuple, st::NamedTuple, batch)
    return Gauss_loss(loss_params, model, ps, st, batch)
end


mutable struct PMMP_Gauss{R, L <: Function} <: LossFunction
    grad_template
    model_param_number::R
    alpha::R
    rho::R
    u_value_multiply_factor::R
    loss_f::L
    fun_p
    fun_pw
    fun_pp
    fun_u
    fun_L2
    L1_alpha::R
    fun1

    function PMMP_Gauss(grad_template, model_param_number; alpha::R=0.1f0, rho::R=0.0f0, u_value_multiply_factor::R=1f0, loss_f::L = Lux.MSELoss(), L1_alpha::R=0f0) where {R, L}
        return new{R, L}(grad_template, model_param_number, alpha, rho, u_value_multiply_factor, loss_f, PMMP_gp_modification, PMMP_gpw_modification, PMMP_gpp_modification, PMMP_u_modification, L2_modification, L1_alpha, L1_PMMP_modification)
    end
end
function (loss_params::PMMP_Gauss)(model, ps::NamedTuple, st::NamedTuple, batch)
    return Gauss_loss(loss_params, model, ps, st, batch)
end