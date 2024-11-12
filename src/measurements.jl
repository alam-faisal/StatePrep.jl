"""
measurements.jl
Author: Faisal Alam
Date: 6/7/2024
defines measurements on matrix product states 
"""

include("mps_utils.jl")

" generates all possible measurement bitstrings "
function outcomes(meas_qubits)
    outcome_tuples = [(0,1) for site_idx in meas_qubits]
    meas_outcomes_list = [x for x in Iterators.product(outcome_tuples...)]
    meas_outcomes_list = reshape(meas_outcomes_list, reduce(*, size(meas_outcomes_list)))
    return meas_outcomes_list
end

" computes the probability of a measurement outcome "
function probability(mps, meas_qubits, meas_outcomes)
    circ = [outcome == 0 ? ("ProjUp", loc) : ("ProjDn", loc) for (loc,outcome) in zip(meas_qubits, meas_outcomes)]
    return abs(inner(mps, runcircuit(mps, circ)))
end

" returns ITensor corresponding to <0| or <1| that contracts against and annihilates an MPS index "
function bra(s, which)
    T = ITensor(s)
    if which == 0
        T[s=>1] = 1.0
        T[s=>2] = 0.0
    else
        T[s=>1] = 0.0
        T[s=>2] = 1.0
    end
    return T
end

" applies list of bra ITensors to the sites of mps listed in meas_qubits, then renames remaining indices using new_inds "
function project(mps, meas_qubits, meas_outcomes; new_inds=nothing, normalized=true)
    mpsc = copy(mps)
    hilbert = siteinds(mpsc)
    n = length(hilbert)
    
    for (loc,outcome) in zip(meas_qubits,meas_outcomes)
        mpsc[loc] = bra(hilbert[loc], outcome) * mpsc[loc] 
    end
    
    new_sites = Vector{ITensor}()
    i = 1
    while i <= n
        if !(i in meas_qubits)
            push!(new_sites, mpsc[i])
            i += 1
        else
            unmeasured = setdiff(i:n, meas_qubits)
            limit = (isempty(unmeasured) ? n : unmeasured[1])

            site = mpsc[i]
            for j in (i+1):limit
                site *= mpsc[j]
            end

            isempty(unmeasured) ? (new_sites[end] *= site) : push!(new_sites, site)
            i = limit + 1
        end
    end
    
    proj_mps = MPS(new_sites)
    proj_mps = normalized ? proj_mps / norm(proj_mps) : proj_mps
    return new_inds === nothing ? proj_mps : rename_indices(proj_mps, new_inds)
end

" measures meas_qubits and returns outcome and post-measurement mps, sampled according to Born probabilities "
function meas_partial(mps, meas_qubits; new_inds=nothing)
    meas_mps = copy(mps)
    
    meas_outcomes = []
    offset = 0
    for loc in meas_qubits
        p = expect(meas_mps,"ProjUp")[loc-offset]
        outcome = rand() < p ? 0 : 1
        meas_mps =  project(meas_mps, [loc-offset], [outcome])  # shouldn't this be loc-offset?
        push!(meas_outcomes, outcome)
        offset += 1
    end
    meas_mps = new_inds === nothing ? meas_mps : rename_indices(meas_mps, new_inds)
    return meas_mps, meas_outcomes
end

" generates measurement samples of mps, sampled according to Born probabilities "
function generate_samples(mps, meas_qubits, num_samples; new_inds=nothing)
    proj_mps_list = []
    meas_outcome_list = []

    for _ in 1:num_samples
        proj_mps, meas_outcome = meas_partial(mps, meas_qubits; new_inds)
        push!(proj_mps_list, proj_mps)
        push!(meas_outcome_list, meas_outcome)
    end

    return proj_mps_list, meas_outcome_list
end

" generates all 2^n measurement outcomes where n is the number of qubits measured " 
function all_samples(mps, meas_qubits; new_inds=nothing, normalized=true)
    outcome_tuples = [(0,1) for site_idx in meas_qubits]
    meas_outcome_list = collect(Iterators.product(outcome_tuples...))
    proj_mps_list = [project(mps, meas_qubits, meas_outcomes; new_inds, normalized) for meas_outcomes in meas_outcome_list]
    return proj_mps_list, meas_outcome_list
end