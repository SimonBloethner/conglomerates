#!/bin/bash
#SBATCH --job-name=id_c42
#SBATCH --output=logs/id_c42_%A_%a.out
#SBATCH --error=logs/id_c42_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# C42: the identity reruns of the existing pilot D scenarios on the C42 code, CHUNK scenarios per array task
# (the per-user limit is 1,000 queued tasks). Array task k runs lines k*CHUNK+1 .. (k+1)*CHUNK of <id list>.
# Usage, from the C42 tree on /scratch:  sbatch --array=0-<ceil(n/CHUNK)-1> submit_c42_identity.sh <id list> <out dir> [CHUNK]

cd "$SLURM_SUBMIT_DIR"
module load miniconda/py312_25.1.1-2
CHUNK=${3:-4}

for j in $(seq 1 "$CHUNK"); do
    LINE=$((SLURM_ARRAY_TASK_ID * CHUNK + j))
    SCENARIO_IDX=$(sed -n "${LINE}p" "$1")
    [ -z "$SCENARIO_IDX" ] && continue
    python run_pilot_c.py $SCENARIO_IDX --pilot d --out "$2" || exit 1  # a failure must show in sacct
    echo "Scenario $SCENARIO_IDX complete"
done
