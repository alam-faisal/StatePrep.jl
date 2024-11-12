"""
train_sqc.jl
Author: Faisal Alam
Date: 11/12/2024
functions for training static quantum circuits 
"""

include("gates.jl")
include("environments.jl")
using Flux: train!
using ProgressMeter
using JLD

" generates either the parameters or gates for SQC ansatz "
function generate_sqc_ansatz(targ_mps, n_layers; data="gates")
    params = rand(n_layers, length(targ_mps)-1, 15)
    if data == "gates"
        return kak_circuit(params)
    else
        return params
    end
end

" infidelity of static quantum circuit with targ_mps given circ "
function sqc_cost(circ, targ_mps)
    hilbert = siteinds(targ_mps)
    return 1 - abs(inner(runcircuit(hilbert, circ), targ_mps))
end

" training SQCs with SVD updates "
function sqc_svd(targ_mps, ansatz; num_sweeps=1000, quiet=false)
    right_mps = deepcopy(targ_mps)
    left_mps = siteinds(targ_mps)
    circ = ansatz
    
    cost_list = [sqc_cost(circ, targ_mps)]
    for sweep in 1:num_sweeps
        for i in 1:length(circ)
            env = environment(left_mps, right_mps, circ, i)
            circ = svd_update(env, circ, i)
        end
        c = sqc_cost(circ, targ_mps)
        push!(cost_list, c)
        
        if abs(cost_list[end] - cost_list[end-1])/cost_list[end] < 1e-5
            break
        end
        
        if !quiet && sweep%50 == 0
            println(cost_list[end])
        end
    end
    return cost_list, circ 
end