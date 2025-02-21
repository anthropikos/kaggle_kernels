#!/bin/bash

#SBATCH --account=ucb524_asc1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --partition=aa100
#SBATCH --output=movement_disorder-%j.out

module purge
module load anaconda

PATH_REQUIREMENT_FILE='../environment.yml'

conda env create -f $PATH_REQUIREMENT_FILE

