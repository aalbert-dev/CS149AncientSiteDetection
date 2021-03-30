#!/bin/bash
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --output=train.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1

# conda activate project1
python train.py --root-dir /local_scratch/COSI149B/Project1/