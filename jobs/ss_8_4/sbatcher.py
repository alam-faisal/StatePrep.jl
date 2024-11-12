import os
import sys

def generate_and_submit_sbatch(line):
    targ_type, n, meas_qubits, n_layers, num_inits, instance = line.split()
    sbatch_file = f"{meas_qubits}.sbatch"
    job_name = f"{targ_type}_{n}_{meas_qubits}"
    
    sbatch_content = f""" 
#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --qos=long
conda activate qaravan
julia cli.jl {line.strip()}
"""

    with open(sbatch_file, 'w') as file:
        file.write(sbatch_content)
    
    os.system(f'sbatch {sbatch_file}')
    print(f"Submitted job '{job_name}' with sbatch file '{sbatch_file}'.")
        
if __name__ == "__main__":
    file_path = sys.argv[1]
    with open(file_path, 'r') as f:
        for line in f:
            generate_and_submit_sbatch(line)