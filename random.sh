#!/bin/bash
#SBATCH -c 24  # Number of Cores per Task
#SBATCH --mem=250000  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --gres=gpu:3  # Number of GPUs
#SBATCH -t 72:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END

source /home/ppjain_umass_edu/miniconda3/etc/profile.d/conda.sh #/home/ppjain_umass_edu/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/etc/profile.d/conda.sh
conda activate imu2clip
