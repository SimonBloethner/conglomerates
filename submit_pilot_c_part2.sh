#!/bin/bash
#SBATCH --job-name=pilot_c2
#SBATCH --output=pilot_c/logs/pilot_c_%A_%a.out
#SBATCH --error=pilot_c/logs/pilot_c_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-274

# Phase C pilot: Part 2 of 2 (scenarios 1000-1274)
# Uses offset: actual scenario = 1000 + array_idx

cd /groups/m-larch/bt307958/IOxEE/programs

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

# Create directories
mkdir -p pilot_c/results
mkdir -p pilot_c/logs

# Compute actual scenario index with offset
SCENARIO_IDX=$((1000 + SLURM_ARRAY_TASK_ID))

# Run single scenario by computed index
python run_pilot_c.py $SCENARIO_IDX

echo "Scenario $SCENARIO_IDX complete"
