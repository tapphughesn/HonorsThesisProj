#!/bin/bash

## This is an example of an sbatch script to run a tensorflow script
## using Singularity to run the tensorflow image.
##
## Set the DATA_PATH to the directory you want the job to run in.
##
## On the singularity command line, replace ./test.py with your program
##
## Change reserved resources as needed for your job.
##

#SBATCH --job-name=seg_metrics
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --partition=general
#SBATCH --output=/proj/NIRAL/users/nic98/processing_scripts/evaluation/seg_metrics-%j.log
#SBATCH --mail-type=end
#SBATCH --mail-user=nic98@live.unc.edu

unset OMP_NUM_THREADS

bash -c "source ~/.bashrc; source activate venv1; python3 /proj/NIRAL/users/nic98/processing_scripts/evaluation/seg_metrics.py"
