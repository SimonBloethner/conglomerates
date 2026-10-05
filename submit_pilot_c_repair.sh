#!/bin/bash
#SBATCH --job-name=pilot_c_repair
#SBATCH --output=pilot_c/logs/repair_%A_%a.out
#SBATCH --error=pilot_c/logs/repair_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-199

# Phase C pilot repair: run missing scenarios from list

cd /groups/m-larch/bt307958/IOxEE/programs

# Load miniconda module
module load miniconda/py312_25.1.1-2

# Read scenario index from missing list (already stripped of leading zeros)
SCENARIO_IDX=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" pilot_c/missing.txt)

# Skip if already exists
if [ -f "pilot_c/results/scenario_$(printf '%04d' $SCENARIO_IDX).pkl" ]; then
    echo "Scenario $SCENARIO_IDX already exists, skipping"
    exit 0
fi

# Run scenario
python run_pilot_c.py $SCENARIO_IDX

echo "Scenario $SCENARIO_IDX complete"
