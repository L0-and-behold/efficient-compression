# The following file contains helper functions for operations involving objects of type NamedTuple. In julia's Lux library the TrainState and neural network parameters are all NamedTuples. 

## Naming convention of apply_fun!:
# The first NamedTuple is the one that is going to be modified by apply_fun!
# The function, that is doing the modification must appear, as argument of apply_fun!, before any of the NamedTuple's it calls
# For example: 
# 1) "apply_fun!(fun::Function, ps::NamedTuple)" applies fun to ps, and writes the result to ps
# 2) apply_fun!(fun::Function, ps1::NamedTuple, ps2::NamedTuple) applies fun to (ps1,ps2), and writes the result to ps1
# 3) apply_fun!(ps1::NamedTuple, fun::Function, ps2::NamedTuple, ps3::NamedTuple) applies fun to (ps2,ps3), and writes the result to ps1 ...

using Revise

function apply_fun!(fun::Function, ps1::NamedTuple)
    for l1 in ps1
        if isa(l1,NamedTuple)
            apply_fun!(fun, l1)
        else
            l1 .= fun.(l1)
        end
    end
end
function apply_fun!(fun::Function, ps1::NamedTuple, ps2::NamedTuple)
    for (l1,l2) in zip(ps1,ps2)
        if isa(l1,NamedTuple)
            apply_fun!(fun, l1, l2)
        else
            l1 .= fun.(l1,l2)
        end
    end
end
function apply_fun!(ps1::NamedTuple, fun::Function, ps2::NamedTuple)
    for (l1,l2) in zip(ps1,ps2)
        if isa(l1,NamedTuple)
            apply_fun!(l1, fun, l2)
        else
            l1 .= fun.(l2)
        end
    end
end
function apply_fun!(ps1::NamedTuple, fun::Function, ps2::NamedTuple, ps3::NamedTuple)
    for (l1,l2,l3) in zip(ps1,ps2,ps3)
        if isa(l1,NamedTuple)
            apply_fun!(l1, fun, l2, l3)
        else
            l1 .= fun.(l2,l3)
        end
    end
end
function apply_fun!_and_sum_nonzero(ps1::NamedTuple, fun::Function, ps2::NamedTuple, ps3::NamedTuple, dtype)
    tsum = zero(dtype)
    for (l1,l2,l3) in zip(ps1,ps2,ps3)
        if isa(l1,NamedTuple)
            apply_fun!_and_sum_nonzero(l1, fun, l2, l3, dtype)
        else
            l1 .= fun.(l2,l3)
            tsum += count(l1.!=0)
        end
    end
    return tsum
end
function L0(ps1::NamedTuple, dtype)
    tsum = zero(dtype)
    for l1 in ps1
        if isa(l1,NamedTuple)
            tsum += L0(l1::NamedTuple, dtype)
        else
            return count(l1.!=0)
        end
    end
    return tsum
end
function check_if_active_set_is_unchanged(active_set, prev_active_set)
    for (l1,l2) in zip(active_set,prev_active_set)
        if isa(l1,NamedTuple)
            return check_if_active_set_is_unchanged(l1, l2)
        else
            if l1 != l2
                return false
            end
        end
    end
    return true
end
function sum_square(u,dtype)
    ss = zero(dtype)
    for l1 in u
        if isa(l1,NamedTuple)
            sum_square(l1,dtype)
        else
            ss += sum(l1.^2)
        end
    end
    return ss
end
function my_sum_diff(ps1,ps2,dtype)
    s = zero(dtype)
    for (l1,l2) in zip(ps1,ps2)
        if isa(l1,NamedTuple)
            s += my_sum_diff(l1,l2,dtype)
        else
            return sum(abs.(l1.-l2))
        end
    end
    return s
end
function my_sum_diff_square(ps1,ps2,dtype)
    s = zero(dtype)
    for (l1,l2) in zip(ps1,ps2)
        if isa(l1,NamedTuple)
            s += my_sum_diff_square(l1,l2,dtype)
        else
            return sum((l1.-l2).^2)
        end
    end
    return s
end
function apply_fun!(fun::Function, ps1::NamedTuple, ps2::NamedTuple, ps3::NamedTuple)
    for (l1,l2,l3) in zip(ps1,ps2,ps3)
        if isa(l1,NamedTuple)
            apply_fun!(fun, l1, l2, l3)
        else
            l1 .= fun.(l1,l2,l3)
        end
    end
end
function apply_fun!_and_sum_square(fun::Function, ps1::NamedTuple, ps2::NamedTuple, ps3::NamedTuple, dtype)
    uss = zero(dtype)
    for (l1,l2,l3) in zip(ps1,ps2,ps3)
        if isa(l1,NamedTuple)
            apply_fun!(fun, l1, l2, l3)
        else
            l1 .= fun.(l1,l2,l3)
            uss += sum(l1.^2)
        end
    end
    return uss
end

function zero_non_contributing_weights!(ps::NamedTuple, r_grads::NamedTuple)
    for (l1,l2) in zip(ps, r_grads)
        if isa(l1,NamedTuple)
            zero_non_contributing_weights!(l1,l2)
        else
            if !isnothing(l2)
                l1[l2 .== 0] .= 0
            end
        end
    end
end
function nested_namedtuple(nt::NamedTuple, func::Function; booltype=false)
    NamedTuple{keys(nt)}(
        (isa(value, NamedTuple) ? nested_namedtuple(value, func; booltype=booltype) : ( booltype ? func.(value) .!= 1.0f0 : func.(value) ) )
        for value in nt
    )
end