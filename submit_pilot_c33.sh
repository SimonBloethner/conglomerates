#!/bin/bash
#SBATCH --job-name=pilot_c33
#SBATCH --output=logs/pilot_c33_%A_%a.out
#SBATCH --error=logs/pilot_c33_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-179

# C33: cost-level-full block, scenarios 1620-1799 (~4 min each)
# Uses offset: actual scenario = 1620 + array_idx (MaxArraySize is 1001)

cd /scratch/bt307958/c33/IOxEE

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

SCENARIO_IDX=$((1620 + SLURM_ARRAY_TASK_ID))

python run_pilot_c.py $SCENARIO_IDX

echo "Scenario $SCENARIO_IDX complete"
