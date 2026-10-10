#!/bin/bash
#SBATCH --job-name=traces_c27
#SBATCH --partition=normal
#SBATCH --array=0-26
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=pilot_c/logs/traces_c27_%A_%a.out
#SBATCH --error=pilot_c/logs/traces_c27_%A_%a.err

# C27: one traced scenario per array task; submit from the repository root.
# Afterwards, on the login node: python analysis/run_traces.py /scratch/$USER/traces_c27 --merge

cd "$SLURM_SUBMIT_DIR"
module load python/3.12.4
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

python analysis/run_traces.py /scratch/$USER/traces_c27 --index $SLURM_ARRAY_TASK_ID
