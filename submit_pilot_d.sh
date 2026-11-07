#!/bin/bash
#SBATCH --job-name=pilot_d
#SBATCH --output=logs/pilot_d_%A_%a.out
#SBATCH --error=logs/pilot_d_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# C37: list-driven pilot D runs (moment rule). C36 moment runtimes: ~5 min on average, search p = 1 and endogenous
# alpha longer; the size block's 100 x 100 runs are the longest.
# Usage, from the C37 tree on /scratch:  sbatch --array=0-<n-1> submit_pilot_d.sh <id list> [<out dir>]
# (<id list>: one pilot D scenario id per line; <out dir> defaults to pilot_d/results.)
# Array task k runs the scenario on line k+1 (MaxArraySize is 1001, so ids are not used as array indices).

cd "$SLURM_SUBMIT_DIR"
module load miniconda/py312_25.1.1-2

SCENARIO_IDX=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")

python run_pilot_c.py $SCENARIO_IDX --pilot d --out "${2:-pilot_d/results}" || exit 1  # a failure must show in sacct

echo "Scenario $SCENARIO_IDX complete"
