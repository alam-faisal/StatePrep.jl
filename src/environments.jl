"""
environments.jl
Author: Faisal Alam
Date: 6/7/2024
environments of parametrized gates 
"""

include("gates.jl")

" environments of one or two qubit gates "
function environment(left_mps, right_mps, circ, i)
    a_mps = run(deepcopy(left_mps), circ[1:i-1])
    b_mps = run(deepcopy(right_mps), circ[i+1:end]; cc=true)
    
    hilbert = siteinds(b_mps) 
    for loc in circ[i][2]
        ind = hilbert[loc]
        replaceind!(b_mps[loc], ind, prime(ind))
    end
    env = contract_all(a_mps, conj.(b_mps))
    return length(circ[i][2]) == 2 ? reshape(permutedims(array(env),[3,1,4,2]),4,4) : array(env)
end

" uses environments to update gate "   
function svd_update(env, circ, gate_idx)
    U,S,V = svd(env)
    new_gate = V * U'
    circ[gate_idx] = ("gate_from_mat", circ[gate_idx][2], (mat=new_gate,))
    return circ 
end