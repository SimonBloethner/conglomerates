#!/bin/bash
#SBATCH --job-name=pilot_c35
#SBATCH --output=logs/pilot_c35_%A_%a.out
#SBATCH --error=logs/pilot_c35_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# C35: list-driven pilot runs (C34 runtimes: 2-5 min, search p = 1 ~8-10 min, endogenous-alpha ~15 min).
# Usage, from the C35 tree on /scratch:  sbatch --array=0-<n-1> submit_pilot_c35.sh <id list> [<out dir>]
# (<id list>: one scenario id per line; <out dir> defaults to pilot_c/results.)
# Array task k runs the scenario on line k+1 (MaxArraySize is 1001, so ids are not used as array indices).

cd "$SLURM_SUBMIT_DIR"

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

SCENARIO_IDX=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")

python run_pilot_c.py $SCENARIO_IDX --out "${2:-pilot_c/results}" || exit 1  # a failed run must show as FAILED in sacct

echo "Scenario $SCENARIO_IDX complete"
