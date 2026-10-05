#!/bin/bash
#SBATCH --job-name=burn_in_c
#SBATCH --output=diagnostics/burn_in_c_%j.out
#SBATCH --error=diagnostics/burn_in_c_%j.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Phase C burn-in analysis
# Runs 5 scenarios with T=8000 to determine convergence

cd /groups/m-larch/bt307958/IOxEE/programs

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

# Create directories
mkdir -p pilot_c
mkdir -p diagnostics

# Run burn-in analysis
python burn_in_analysis.py

echo "Burn-in analysis complete"
