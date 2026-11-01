#!/bin/bash
#SBATCH --job-name=burnin_c33_ext
#SBATCH --output=logs/burnin_c33_ext_%A_%a.out
#SBATCH --error=logs/burnin_c33_ext_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-19

# C33: burn-in runs extended to T=16000 (~5 min each). The model is single-threaded, so one core per run.

cd /scratch/bt307958/c33/IOxEE

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

python run_burnin_c33_ext.py $SLURM_ARRAY_TASK_ID

echo "Run $SLURM_ARRAY_TASK_ID complete"
