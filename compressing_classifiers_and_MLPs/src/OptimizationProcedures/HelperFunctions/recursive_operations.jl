using CUDA, Revise
include("loss_functions.jl")

# The following file contains recursive update operations that are applied to objects of type NamedTuple.
# They are used by update_functions.jl to modify gradients of parameters explicitly without the need to do backpropagation.
# Such explicit gradients arise in the regularization procedures we employ (like DRR, PMMP or RL1) and are more efficient than their corresponding backprop counterparts.

function recursively_modify!(params1, params2, loss_fun, fun)
    if !isnothing(params1)
        for (subparams1, subparams2) in zip(params1, params2)
            if isa(subparams1, AbstractArray{T} where T)
                subparams1 .+= fun(loss_fun, subparams1, subparams2, params2)
            else
                recursively_modify!(subparams1, subparams2, loss_fun, fun)
            end
        end
    end
end
function recursively_reassign!(params1, params2, loss_fun, fun)
    if !isnothing(params1)
        for (subparams1, subparams2) in zip(params1, params2)
            if isa(subparams1, AbstractArray{T} where T)
                subparams1 .= fun(loss_fun, subparams1, subparams2, params2)
            else
                recursively_reassign!(subparams1, subparams2, loss_fun, fun)
            end
        end
    end
end

function recursively_modify_DRR!(params1, params2, loss_fun::DRR, fun)
    if !isnothing(params1)
        for (subparams1, (name_subparam2,subparams2)) in zip(params1, pairs(params2))
            if startswith(string(name_subparam2), "layer_")
                loss_fun.layer_param_number .= Lux.parameterlength(subparams2)
            end
            if isa(subparams1, AbstractArray{T} where T)
                subparams1 .+= fun(loss_fun, subparams1, subparams2, params2)
            else
                recursively_modify_DRR!(subparams1, subparams2, loss_fun, fun)
            end
        end
    end
end

function recursively_modify_PMMP!(grads_p, p, pw, pp, u, gp, gpw, gpp, gu, loss_fun::Union{PMMP, PMMP_Gauss}, fun_p, fun_pw, fun_pp, fun_u)
    if !isnothing(grads_p)
        for (sub_grads_p, sub_p, sub_pw, sub_pp, sub_u, sub_gp, sub_gpw, sub_gpp, sub_gu) in zip(grads_p, p, pw, pp, u, gp, gpw, gpp, gu)
            if isa(sub_p, AbstractArray{T} where T)
                fun_p(sub_gp, sub_grads_p, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
                fun_pw(sub_gpw, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
                fun_pp(sub_gpp, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
                fun_u(sub_gu, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
            else
                recursively_modify_PMMP!(sub_grads_p, sub_p, sub_pw, sub_pp, sub_u, sub_gp, sub_gpw, sub_gpp, sub_gu, loss_fun, fun_p, fun_pw, fun_pp, fun_u)
            end
        end
    end
end
function recursively_modify_FPP!(grads_p, p, pw, pp, u, gp, gpw, gpp, loss_fun::Union{FPP, FPP_Gauss}, fun_p, fun_pw, fun_pp)
    if !isnothing(grads_p)
        for (sub_grads_p, sub_p, sub_pw, sub_pp, sub_u, sub_gp, sub_gpw, sub_gpp) in zip(grads_p, p, pw, pp, u, gp, gpw, gpp)
            if isa(sub_p, AbstractArray{T} where T)
                fun_p(sub_gp, sub_grads_p, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
                fun_pw(sub_gpw, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
                fun_pp(sub_gpp, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
            else
                recursively_modify_FPP!(sub_grads_p, sub_p, sub_pw, sub_pp, sub_u, sub_gp, sub_gpw, sub_gpp, loss_fun, fun_p, fun_pw, fun_pp)
            end
        end
    end
end
function recursively_modify_FPP_u!(p, pw, pp, u, gu, loss_fun::Union{FPP, FPP_Gauss}, fun_u)
    if !isnothing(gu)
        for (sub_p, sub_pw, sub_pp, sub_u, sub_gu) in zip(p, pw, pp, u, gu)
            if isa(sub_p, AbstractArray{T} where T)
                fun_u(sub_gu, sub_p, sub_pw, sub_pp, sub_u, loss_fun)
            else
                recursively_modify_FPP_u!(sub_p, sub_pw, sub_pp, sub_u, sub_gu, loss_fun, fun_u)
            end
        end
    end
end

function project_params!(params)
    if !isnothing(params)
        for subparams in params
            if isa(subparams, AbstractArray{T} where T)
                clamp!(subparams, 0, 1) # this is curiously 10x faster than the two lines below on GPU but almost equivalent performance on CPU
                # subparams[subparams .< 0] .= 0
                # subparams[subparams .> 1] .= 1
            else
                project_params!(subparams)
            end
        end
    end
end

