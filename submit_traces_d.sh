#!/bin/bash
#SBATCH --job-name=traces_d
#SBATCH --partition=normal
#SBATCH --array=0-26
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=10:00:00
#SBATCH --output=logs/traces_d_%A_%a.out
#SBATCH --error=logs/traces_d_%A_%a.err

# C37: the 27 traced pilot D runs (main, power law, 3 families x alpha {0.05, 0.1, 0.3} x reps 0-2, moment rule),
# one per array task: the traced run and its event study, then the status panel and the floor / no-floor
# decomposition on the same trace. Submit from the C37 tree after pilot_d/results holds the copied C36 runs
# (the decomposition checks the trace against the result pickle). Merge afterwards (final_c37.sh).

cd "$SLURM_SUBMIT_DIR"
module load miniconda/py312_25.1.1-2
K=$SLURM_ARRAY_TASK_ID

python analysis/run_traces.py pilot_d/traces --index $K --pilot d || exit 1
python analysis/run_panel.py pilot_d/traces --index $K --pilot d || exit 1
python analysis/run_decomposition.py pilot_d/traces --index $K --pilot d || exit 1
echo "Trace $K complete"
