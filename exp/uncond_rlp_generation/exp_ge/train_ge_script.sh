#!/bin/bash
#
#SBATCH --job-name="EM-trans"
#SBATCH --partition=gpu-a100
#SBATCH --time=99:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=research-eemcs-ese

module load 2023r1 
module load cuda/11.6

srun exp/uncond_rlp_generation/exp_ge/train_ge.py
