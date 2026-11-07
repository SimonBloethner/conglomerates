#!/bin/bash
#SBATCH --job-name=c38_traces
#SBATCH --partition=normal
#SBATCH --array=0-26
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/c38_traces_%A_%a.out
#SBATCH --error=logs/c38_traces_%A_%a.err

# C38 Task 1: the event study's alternative matchings on the 27 existing pilot D traces (no model run), one trace
# per array task -> pilot_d/events/<tag>_matching.csv. Merge afterwards:
#   python analysis/run_traces.py pilot_d/traces --merge-matching --pilot d

cd "$SLURM_SUBMIT_DIR"
module load miniconda/py312_25.1.1-2

python analysis/run_traces.py pilot_d/traces --matching --index $SLURM_ARRAY_TASK_ID --pilot d || exit 1
echo "Trace $SLURM_ARRAY_TASK_ID complete"
