#!/bin/bash
#SBATCH -J dft_ml
#SBATCH -o %j-learning-dft.out
#SBATCH -e %j-learning-dft.err
#SBATCH --mem=200G
#SBATCH -p development 
#SBATCH --gres=gpu:1

module purge
module load python/3
module load tensorflow/1.2_gpu


python3 learn_densities.py
