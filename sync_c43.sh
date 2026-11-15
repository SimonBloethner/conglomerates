#!/bin/bash
#SBATCH --job-name=c43_sync
#SBATCH --output=logs/c43_sync_%j.out
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --mem=1G
# C43: copy the 45 alpha-gap pickles to /groups (dd oflag=sync: buffered writes > 1 MiB to /groups fail) and verify
# by md5. Usage: sbatch --dependency=afterok:<array job> sync_c43.sh
cd "$SLURM_SUBMIT_DIR"
DST=/groups/m-larch/bt307958/IOxEE_c43/pilot_d/results_c43
mkdir -p "$DST"
for f in pilot_d/results_c43/*.pkl.gz; do
    dd if="$f" of="$DST/$(basename "$f")" bs=64k oflag=sync status=none
done
(cd pilot_d/results_c43 && md5sum *.pkl.gz) > /tmp/c43_src.md5
(cd "$DST" && md5sum *.pkl.gz) > /tmp/c43_dst.md5
if cmp -s /tmp/c43_src.md5 /tmp/c43_dst.md5; then echo "SYNC OK: $(wc -l < /tmp/c43_src.md5) files"; else echo "SYNC MISMATCH"; diff /tmp/c43_src.md5 /tmp/c43_dst.md5; exit 1; fi
