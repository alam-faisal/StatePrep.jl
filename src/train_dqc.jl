"""
train_dqc.jl
Author: Faisal Alam
Date: 11/12/2024
functions for training dynamic quantum circuits 
"""

include("gates.jl")
include("measurements.jl")
include("environments.jl")
using ProgressMeter
using JLD
using Dates

" generates either the parameters or gates for DQC ansatz "
function generate_dqc_ansatz(targ_mps, meas_qubits, n_layers; data="gates", post="Rn", n_post_layers=0)
    n_dynamic = length(targ_mps) + length(meas_qubits)
    hilbert = siteinds("Qubit", n_dynamic)
    active_qubits = [x for x in 1:n_dynamic if !(x in meas_qubits)] 
    
    pre_params = rand(n_layers, n_dynamic-1, 15)
    pre_circ = kak_circuit(pre_params)
    
    if post == "Rn"
        post_params = rand(2^length(meas_qubits), length(targ_mps), 3)
        post_circ_list = [rn_layer(post_params[i,:,:], active_qubits) for i in 1:2^length(meas_qubits)]
    else
        post_params = rand(2^length(meas_qubits), n_post_layers, length(targ_mps)-1, 15)
        post_circ_list = [kak_circuit(post_params[i,:,:,:], active_qubits) for i in 1:2^length(meas_qubits)]
    end
        
    if data == "gates"
        return (hilbert, meas_qubits, active_qubits, pre_circ, post_circ_list)
    else
        return (hilbert, meas_qubits, active_qubits, pre_params, post_params)
    end
end 

" turns target state into an ensemble corresponding to measurement outcomes "
function targ_ensemble(targ_mps, meas_qubits, hilbert, post_circ_list)
    dressed_targ_mps = deepcopy(targ_mps)
    for loc in meas_qubits
        dressed_targ_mps = insert_zero_ket(dressed_targ_mps, loc)
    end
    dressed_targ_mps = rename_indices(dressed_targ_mps, hilbert)
    
    meas_outcomes_list = outcomes(meas_qubits)
    ensemble = []
    for meas_outcomes in meas_outcomes_list       
        tmps = deepcopy(dressed_targ_mps)
        proj_circ = [outcome == 0 ? ("I", loc) : ("X", loc) for (loc,outcome) in zip(meas_qubits, meas_outcomes)]
        outcome_int = foldl((x, y) -> 2x + y, reverse(meas_outcomes)) + 1
        post_circ = post_circ_list[outcome_int]
        tmps = runcircuit(tmps, vcat(proj_circ, post_circ)) 
        push!(ensemble, tmps)
    end
    return ensemble
end   

" infidelity averaged over every measurement outcome; takes gates of kak_circuit() and rn_layer()s "
function dqc_cost(pre_circ, post_circ_list, hilbert, meas_qubits, targ_mps) 
    mps = runcircuit(hilbert, pre_circ)
    t_ensemble = targ_ensemble(targ_mps, meas_qubits, hilbert, post_circ_list)
    meas_outcomes_list = outcomes(meas_qubits)
    renorm_list = [sqrt(probability(mps, meas_qubits, meas_outcomes)) for meas_outcomes in meas_outcomes_list]
    infidelity = sum([1 - abs(inner(mps, t_mps)/renorm) for (t_mps, renorm) in zip(t_ensemble,renorm_list)])
    return infidelity / (2^length(meas_qubits))    
end

" infidelity averaged over every measurement outcome; takes gates of kak_circuit() and ensemble of target states "
function dqc_cost(pre_circ, hilbert, meas_qubits, t_ensemble)     
    mps = runcircuit(hilbert, pre_circ)
    meas_outcomes_list = outcomes(meas_qubits)
    renorm_list = [sqrt(probability(mps, meas_qubits, meas_outcomes)) for meas_outcomes in meas_outcomes_list]
    infidelity = sum([1 - abs(inner(mps, t_mps)/renorm) for (t_mps, renorm) in zip(t_ensemble,renorm_list)])
    return infidelity / (2^length(meas_qubits))    
end

" computes environment of a pre-measurement gate in a dynamic quantum circuit "
function dqc_environment(pre_circ, hilbert, meas_qubits, gate_idx, t_ensemble)   
    mps = runcircuit(hilbert, pre_circ)
    meas_outcomes_list = outcomes(meas_qubits)
    renorm_list = [sqrt(probability(mps, meas_qubits, meas_outcomes)) for meas_outcomes in meas_outcomes_list]
    env = sum([environment(hilbert, t_mps, pre_circ, gate_idx)/renorm for (t_mps,renorm) in zip(t_ensemble,renorm_list)])
    return env
end

" optimizes pre-measurement circuit using SVD updates "
function optim_pre_circ(pre_circ, hilbert, meas_qubits, t_ensemble, num_sweeps=2)
    for sweep in 1:num_sweeps
        for gate_idx in 1:length(pre_circ)
            env = dqc_environment(pre_circ, hilbert, meas_qubits, gate_idx, t_ensemble)
            pre_circ = svd_update(env, pre_circ, gate_idx)
        end
    end
    return pre_circ
end

" optimizes post-measurement circuit using SVD updates "
function optim_post_circ(post_circ, proj_mps, targ_mps, active_qubits, num_sweeps=2)
    post_circ = rn_layer_adjoint(post_circ, 1:length(targ_mps))
    for sweep in 1:num_sweeps
        for gate_idx in 1:length(post_circ)
            env = environment(proj_mps, targ_mps, post_circ, gate_idx)
            post_circ = svd_update(env, post_circ, gate_idx)
        end
    end
    return rn_layer_adjoint(post_circ, active_qubits)
end

" trains a dynamic quantum circuit using svd updates "
function dqc_svd(targ_mps, ansatz; num_sweeps=5000, quiet=false)
    start_time = now()
    (hilbert, meas_qubits, active_qubits, pre_circ, post_circ_list) = ansatz
    
    cost_list = [dqc_cost(pre_circ, post_circ_list, hilbert, meas_qubits, targ_mps)]
    time_list = [Millisecond(0)]
    for epoch in 1:num_sweeps
        t_ensemble = targ_ensemble(targ_mps, meas_qubits, hilbert, post_circ_list)
        pre_circ = optim_pre_circ(pre_circ, hilbert, meas_qubits, t_ensemble)
        c = dqc_cost(pre_circ, hilbert, meas_qubits, t_ensemble) 
        push!(cost_list, c)
        push!(time_list, now()-start_time)
        
        mps = runcircuit(hilbert, pre_circ)
        proj_mps_list, meas_outcomes_list = all_samples(mps, meas_qubits; new_inds=siteinds(targ_mps))
        
        for (k,(proj_mps,post_circ)) in enumerate(zip(proj_mps_list, post_circ_list))
            post_circ = optim_post_circ(post_circ, proj_mps, targ_mps, active_qubits)
            post_circ_list[k] = post_circ
        end
        
        t_ensemble = targ_ensemble(targ_mps, meas_qubits, hilbert, post_circ_list)
        c = dqc_cost(pre_circ, hilbert, meas_qubits, t_ensemble) 
        push!(cost_list, c)
        push!(time_list, now()-start_time)
        
        if abs(cost_list[end] - cost_list[end-2])/cost_list[end] < 1e-5
            break
        end
        
        if !quiet && epoch%10 == 0
            println(c)
        end
    end
    return cost_list, pre_circ, post_circ_list, time_list
end