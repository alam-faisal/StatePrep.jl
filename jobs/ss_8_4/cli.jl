"""
cli.jl
Author: Faisal Alam
Date: 11/12/2024
command line interface for running jobs
"""
include("../../src/models.jl")
include("../../src/sqc_train.jl")
include("../../src/dqc_train.jl")


targ_type, n, meas_qubits, n_layers, num_inits, instance = ARGS
job_id = "$(targ_type)_$n"
println("Starting simulation for job_id: $job_id")

error_file = "error_$(job_id).err"
mkpath(dirname(error_file))

n = parse(Int, n)
meas_qubits = eval(Meta.parse(meas_qubits))
n_layers = parse(Int, n_layers)
num_inits = parse(Int, num_inits)
instance = parse(Int, instance)

targfile = "$(targ_type)_$n_$(instance).jld2"
println("Loading target MPS from $targfile")
targ_mps = load(targfile, "targ_mps")

outfile = "$(targ_type)_$n_$(instance)_$(n_layers)_$(meas_qubits).jld2"
if !isnothing(instance)
    push!(outfile, instance)
end
outfile = joinpath(directory, join(outfile, "_") * ".jld")

results = []
for init in 1:num_inits   
    println("Running initialization $init of $num_inits for instance: $instance")
    if length(meas_qubits) > 0
        ansatz = generate_dqc_ansatz(targ_mps, meas_qubits, n_layers)
        result = dqc_svd(targ_mps, ansatz)
    else
        ansatz = generate_sqc_ansatz(targ_mps, n_layers)
        result = sqc_svd(targ_mps, ansatz)   
    end
    push!(results, result)
end

println("Saving results to $outfile for job_id: $job_id")
save(outfile, "results", results)
println("Completed simulation for job_id: $job_id")