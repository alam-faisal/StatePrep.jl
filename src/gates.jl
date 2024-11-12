"""
gates.jl
Author: Faisal Alam
Date: 6/7/2024
Defines custom gates and circuits not included in PastaQ
"""

using ITensors
using PastaQ
using LinearAlgebra

x = array(op("X", siteind("Qubit")))
y = array(op("Y", siteind("Qubit")))
z = array(op("Z", siteind("Qubit")))
xx = kron(x,x)
yy = kron(y,y)
zz = kron(z,z)

" creates ITensor operator for the identity matrix " 
function ITensors.op(::OpName"I", ::SiteType"Qubit")
    return [1 0 
            0 1]
end

" creates ITensors operator from given unitary matrix "
function ITensors.op(::OpName"gate_from_mat", ::SiteType"Qubit"; mat)
    return mat
end

" returns the adjoint of a gate_from_mat "
function adjoint(gate)
    return ("gate_from_mat", gate[2], (mat=copy(gate[3].mat'),))
end

" given 3 angles, returns Rn gate in matrix form; size(params) = (3,) "
function rn_matrix(params)
    return [cos(params[1]/2) -exp(1im*params[3])*sin(params[1]/2)
            exp(1im*params[2])*sin(params[1]/2) exp(1im*(params[3]+params[2]))*cos(params[1]/2)]
end

" layer of Rn gates; size(params) = (n,3) "
function rn_layer(params, qubits) 
    return [("gate_from_mat", loc, (mat=rn_matrix(params[j,:]),)) for (j,loc) in enumerate(qubits)]
end

" returns the adjoint of a layer of Rn gates and also changes the qubit locations where the gates act "
function rn_layer_adjoint(circ, new_locs)
    n = length(circ)
    return [("gate_from_mat", loc, (mat=copy(circ[i][3].mat'),)) for (loc,i) in zip(new_locs,1:n)]
end    

" given 15 angles, returns KAK gate in matrix form; size(params) = (15,) "
function kak_matrix(params)
    left = kron(rn_matrix(params[1:3]), rn_matrix(params[4:6]))
    arg = params[7] * xx + params[8] * yy + params[9] * zz
    mid = exp(1im * arg)
    right = kron(rn_matrix(params[10:12]), rn_matrix(params[13:15]))
    return left * mid * right
end

" bricklayer of KAK gates; size(params) = (n-1,15) "
function kak_layer(params, qubits=nothing)
    n = size(params)[1] + 1
    qubits = qubits === nothing ? range(1, stop=n) : qubits
    circ = []
    for i in 1:n-1
        mat = kak_matrix(params[i,:])
        loc = Int(2 * ((i-1) % div(n,2)) + 1 + (i > (n/2)))
        gate = ("gate_from_mat", (qubits[loc], qubits[loc+1]), (mat=mat,))
        circ = [circ; gate]
    end
    return circ
end   

" several bricklayers of KAK gates; size(params) = (n_layers,n-1,15) "
function kak_circuit(params, qubits=nothing)
    n_layers = size(params)[1]
    circ = []
    for i in 1:n_layers
        layer_circ = kak_layer(params[i,:,:], qubits)
        circ = [circ; layer_circ]
    end
    return circ
end

" modifies PastaQ's runcircuit function to include possibility of adjointing the circuit "
function run(mps, circ; cc=false)
    circ = cc ? [adjoint(g) for g in reverse(circ)] : circ
    return runcircuit(mps, circ)
end