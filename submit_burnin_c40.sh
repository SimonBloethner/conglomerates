#!/bin/bash
#SBATCH --job-name=burnin_c40
#SBATCH --output=logs/burnin_c40_%A_%a.out
#SBATCH --error=logs/burnin_c40_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-19

# C40: the C33 burn-in runs under the moment rule (pilot D), T=16000 (~5 min each). One core per run.
# A run that does not reproduce its pilot D result up to step 11000 exits with code 2.

cd /scratch/bt307958/c40/IOxEE

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

python run_burnin_c40.py $SLURM_ARRAY_TASK_ID

echo "Run $SLURM_ARRAY_TASK_ID complete"
