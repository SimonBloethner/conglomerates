#!/bin/bash
#SBATCH --job-name=phase_a_diagnostics
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/groups/m-larch/bt307958/conglomerates_dev/diagnostics/diag_%j.out
#SBATCH --error=/groups/m-larch/bt307958/conglomerates_dev/diagnostics/diag_%j.err

# Load modules
module load python/3.13.3

# Go to work directory
cd /groups/m-larch/bt307958/conglomerates_dev

# Create output dir
mkdir -p diagnostics

# Activate environment
source cong_env/bin/activate

echo "Starting diagnostic run at $(date)"
echo "Python: $(which python)"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"

# Run diagnostics
python run_diagnostics.py

echo "Completed at $(date)"
