#!/bin/bash
#SBATCH --job-name=JPO25_mu1p0_EKE_h0s8_10_12  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G         # memory per cpu-core (4G is default)
#SBATCH --time=47:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ml1994@princeton.edu

module purge
module load julia/1.9.1

julia compute_mu1p0_wf_spec_h1012.jl
