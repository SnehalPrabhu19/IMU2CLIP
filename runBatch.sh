#!/bin/bash
#SBATCH -c 24  # Number of Cores per Task
#SBATCH --mem=250000  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --gres=gpu:3  # Number of GPUs
#SBATCH -t 72:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END

source /home/ppjain_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate imu2clip

# cd /home/ppjain_umass_edu/AmbientAI_IMU2CLIP/
# bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_patchrnn.yaml
# bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_patchtransformer.yaml
# bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_attentionpooled.yaml
bash runModel.sh /home/ppjain_umass_edu/AmbientAI_IMU2CLIP/configs/train_contrastive/ego4d_imu2text_mw2.yaml

# cd /home/pranayr_umass_edu/meta_project/imu2clip/
# bash download_videos.sh
# bash download_clips.sh