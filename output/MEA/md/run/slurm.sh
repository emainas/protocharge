#!/usr/bin/env bash
#SBATCH -J MEA-md
#SBATCH -p l40-gpu
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

bash run.sh
