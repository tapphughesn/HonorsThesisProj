#!/bin/bash

## SLURM script using a singularity container with tensorflow-gpu 2.2.0 to train

#SBATCH --job-name=make_boundary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --partition=general
#SBATCH --mail-type=end
#SBATCH --mail-user=nic98@live.unc.edu

unset OMP_NUM_THREADS
source activate venv1;
python3 /proj/NIRAL/users/nic98/processing_scripts/preprocessing/boundary_training_arrays.py;

