#!/bin/bash
#SBATCH --job-name=pilot_c34
#SBATCH --output=logs/pilot_c34_%A_%a.out
#SBATCH --error=logs/pilot_c34_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# C34: list-driven pilot runs (C33 runtimes: 2-5 min, endogenous-alpha ~17 min, longest 18 min).
# Usage, from the C34 tree on /scratch:  sbatch --array=0-<n-1> submit_pilot_c34.sh <file with one scenario id per line>
# Array task k runs the scenario on line k+1 (MaxArraySize is 1001, so ids are not used as array indices).

cd "$SLURM_SUBMIT_DIR"

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

SCENARIO_IDX=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")

python run_pilot_c.py $SCENARIO_IDX || exit 1  # a failed run must show as FAILED in sacct

echo "Scenario $SCENARIO_IDX complete"
