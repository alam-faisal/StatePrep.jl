""" 
models.jl
Author: Faisal Alam
Date: 11/12/2024
paradigmatic classes of states to use as targets for DQCs
"""

include("gates.jl")

" generates a n_clusters product of GHZ states of size cluster_size "  
function ghz_clusters(cluster_size, n_clusters)
    circuit = []
    for j in 1:n_clusters
        start = 1 + cluster_size*(j-1)
        circuit = [circuit; ("H", start)]
        for i in start:start+cluster_size-2
            circuit = [circuit; ("CX", (i,i+1))]
        end
    end
    
    for j in 1:n_clusters-1
        fused = j*cluster_size
        circuit = [circuit; ("CX", (fused,fused+1))]
        circuit = [circuit; ("H", fused)]
    end
    
    hilbert = qubits(cluster_size*n_clusters)
    mps = runcircuit(hilbert, circuit)
    return mps
end

" returns ground state of the critical transverse-field Ising model " 
function tfim_gs(N)
    sites = siteinds("Qubit",N)
    os = OpSum()
    for j=1:N-1
        os += -4.0,"Sz",j,"Sz",j+1
        os += -2.0,"Sx",j  # set to -2 for critical point
    end
    os += -2.0,"Sx",N   # set to -2 for critical point
    H = MPO(os,sites)
    
    psi0 = randomMPS(sites,N)
    nsweeps = 5
    maxdim = [N,2*N,3*N,4*N,5*N]
    cutoff = [1E-10]
    energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff, outputlevel=0)
    
    return psi
end  

" returns random MPS with bond dimension determined by n_layers "
function rand_mps(n,n_layers) 
    mps = randomMPS(siteinds("Qubit", n); linkdims=2^(2*n_layers-1))
    truncate!(mps; cutoff=1e-14)
    return mps
end

" returns random subset state with subset size given by k "
function subset_state(n, k)
    s = siteinds("Qubit", n)
    mps_list = [(1/sqrt(k)) * MPS(s, string.([rand(0:1) for _ in 1:n])) for i in 1:k]
    return sum(mps_list)
end

" state output by random quantum circuit of depth n_layers "
function generic_state(n,n_layers)
    s = siteinds("Qubit", n)
    params = rand(n_layers, n-1, 15)
    return runcircuit(s, kak_circuit(params))
end

" returns (and writes to file) a target state with specified attributes "
function model_mps(targ_type, n; n_layers=nothing, k=nothing, dir=nothing, instance=nothing)
    if targ_type == "ghz"
        mps = ghz_clusters(n,1)
    elseif targ_type == "tfi"
        mps = tfim_gs(n)
    elseif targ_type == "mps"
        mps = rand_mps(n, n_layers)
    elseif targ_type == "ss"
        mps = subset_state(n, k)
    elseif targ_type == "gen"
        mps = generic_state(n, n_layers)        
    else
        @error "Target state of type $targ_type has not been implemented"
        return 
    end
    
    if !isnothing(dir)
        filename = "$(targ_type)_$n_$(instance).jld2"
        save(joinpath(dir, filename), "targ_mps", mps)
    end
    return mps
end