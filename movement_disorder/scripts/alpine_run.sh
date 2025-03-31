#!/bin/bash

#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --partition=aa100
#SBATCH --output=movement_disorder-%j.out

module purge
module load miniforge

mamba activate movement_disorder

python main.py
