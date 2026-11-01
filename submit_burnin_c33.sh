#!/bin/bash
#SBATCH --job-name=burnin_c33
#SBATCH --output=logs/burnin_c33_%A_%a.out
#SBATCH --error=logs/burnin_c33_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-19

# C33: burn-in diagnostic, 20 pilot runs (~3 min each)

cd /scratch/bt307958/c33/IOxEE

# Load miniconda module (has numba, numpy, scipy)
module load miniconda/py312_25.1.1-2

python run_burnin_c33.py $SLURM_ARRAY_TASK_ID

echo "Run $SLURM_ARRAY_TASK_ID complete"
