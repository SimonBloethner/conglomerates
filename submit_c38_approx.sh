#!/bin/bash
#SBATCH --job-name=c38_approx
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/c38_approx_%j.out
#SBATCH --error=logs/c38_approx_%j.err

# C38 Task 3: the moment rule's approximation error on the 27 pilot D traces -> pilot_d/approximation_error.csv

cd "$SLURM_SUBMIT_DIR"
module load miniconda/py312_25.1.1-2

python analysis/approximation_error.py pilot_d/traces --n 2000 --pilot d || exit 1
