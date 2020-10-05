#!/bin/bash

## SLURM script using a singularity container with tensorflow-gpu 2.2.0 to train

#SBATCH --job-name=pretrain_cont1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --output=/proj/NIRAL/users/nic98/models/unet_4/logfiles/UNET_4_run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=nic98@live.unc.edu

unset OMP_NUM_THREADS

# Set SIMG (Singularity Image) path
SIMG_PATH=/proj/NIRAL/users/nic98/singularity_containers

# Set SIMG name
SIMG_NAME=tf2.2.1_gpu.simg

# Set data path
DATA_PATH=/proj/NIRAL/users/nic98/models/unet_4

# GPU with Singularity
singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "cd $DATA_PATH; python3 small_pretrain.py"

