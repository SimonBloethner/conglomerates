#!/bin/bash
#SBATCH --job-name=pilot_c
#SBATCH --output=pilot_c/logs/pilot_c_%A_%a.out
#SBATCH --error=pilot_c/logs/pilot_c_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-999

# Phase C pilot: Part 1 of 2 (scenarios 0-999)
# Each scenario: T=6000 steps, ~60-90s runtime

cd /groups/m-larch/bt307958/IOxEE/programs

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

# Create directories
mkdir -p pilot_c/results
mkdir -p pilot_c/logs

# Run single scenario by array index
python run_pilot_c.py $SLURM_ARRAY_TASK_ID

echo "Scenario $SLURM_ARRAY_TASK_ID complete"
