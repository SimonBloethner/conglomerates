#!/usr/bin/env python3
"""
Script to run counterfactual analysis across multiple SLURM jobs in parallel
"""

import os
import subprocess
import time
import numpy as np

# Configuration
SHARES = np.arange(0, 0.52, 0.02)  # Share values to test
COUNTERFACTUALS_PER_SHARE = 50
CHUNK_SIZE = 25  # Each job processes 25 experiments per share
N_CHUNKS_PER_SHARE = COUNTERFACTUALS_PER_SHARE // CHUNK_SIZE
CORES_PER_JOB = 32  # Fewer cores per job since we're chunking experiments
TIME_LIMIT = "12:00:00"  # 12 hours per job

def create_directories():
    """Create organized directory structure"""
    directories = ['counterfactual_slurm_logs', 'counterfactual_slurm_scripts', 'counterfactual_results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_slurm_script(share_value, chunk_id, chunk_start, chunk_end):
    """Create a SLURM batch script for a specific share/chunk combination"""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=s{share_value:.2f}_c{chunk_id}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task={CORES_PER_JOB}
#SBATCH --time={TIME_LIMIT}
#SBATCH --output=/groups/m-larch/bt307958/conglomerates/counterfactual_slurm_logs/share_{share_value:.2f}_chunk_{chunk_id}.out
#SBATCH --error=/groups/m-larch/bt307958/conglomerates/counterfactual_slurm_logs/share_{share_value:.2f}_chunk_{chunk_id}.err

# Load required modules
module load python/3.13.3

# Change to the correct directory
cd /groups/m-larch/bt307958/conglomerates

# Activate environment and verify
source cong_env/bin/activate

# Debug environment activation
echo "Python path: $(which python3)"
echo "Environment: $VIRTUAL_ENV"
echo "Virtual env Python: $VIRTUAL_ENV/bin/python"

echo "Testing imports..."
$VIRTUAL_ENV/bin/python -c "import tqdm; import sklearn; print('✅ All imports successful')" || echo "❌ Import failed"

# Run the counterfactual analysis for this share/chunk
echo "Starting share {share_value:.2f}, chunk {chunk_id}: experiments {chunk_start}-{chunk_end}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"

$VIRTUAL_ENV/bin/python parallel_counterfactuals.py --share {share_value:.2f} --chunk {chunk_start}_{chunk_end} --n_cores {CORES_PER_JOB} --counterfactuals {COUNTERFACTUALS_PER_SHARE}

echo "Share {share_value:.2f}, chunk {chunk_id} completed at: $(date)"
"""
    
    script_filename = f"counterfactual_slurm_scripts/job_share_{share_value:.2f}_chunk_{chunk_id}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    
    return script_filename

def submit_jobs():
    """Submit all counterfactual jobs to SLURM"""
    create_directories()
    
    job_ids = []
    total_jobs = 0
    
    for share_value in SHARES:
        for chunk_id in range(N_CHUNKS_PER_SHARE):
            chunk_start = chunk_id * CHUNK_SIZE
            chunk_end = min((chunk_id + 1) * CHUNK_SIZE, COUNTERFACTUALS_PER_SHARE)
            
            # Create SLURM script
            script_file = create_slurm_script(share_value, chunk_id, chunk_start, chunk_end)
            
            # Submit job
            result = subprocess.run(['sbatch', script_file], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"Submitted share {share_value:.2f}, chunk {chunk_id}: Job ID {job_id}")
                total_jobs += 1
            else:
                print(f"Failed to submit share {share_value:.2f}, chunk {chunk_id}: {result.stderr}")
    
    return job_ids, total_jobs

def monitor_jobs(job_ids):
    """Monitor job progress"""
    print(f"\\nMonitoring {len(job_ids)} jobs...")
    print("Use 'squeue -u $USER' to check job status")
    print("Use 'tail -f counterfactual_slurm_logs/*.out' to monitor progress")
    
    # Wait for all jobs to complete
    while True:
        result = subprocess.run(['squeue', '-u', os.getenv('USER'), '--noheader'], 
                              capture_output=True, text=True)
        
        running_jobs = [line.split()[0] for line in result.stdout.strip().split('\\n') if line]
        our_running = [jid for jid in job_ids if jid in running_jobs]
        
        if not our_running:
            print("\\nAll jobs completed!")
            break
        
        print(f"\\rJobs still running: {len(our_running)}/{len(job_ids)}", end='', flush=True)
        time.sleep(60)  # Check every minute

def merge_results():
    """Merge results from all chunks into final counterfactual dataset"""
    import pickle
    
    print("\\nMerging counterfactual results...")
    
    all_share_results = {}
    
    for share_value in SHARES:
        print(f"Merging results for share {share_value:.2f}")
        
        share_experiments = []
        
        for chunk_id in range(N_CHUNKS_PER_SHARE):
            chunk_start = chunk_id * CHUNK_SIZE
            chunk_end = min((chunk_id + 1) * CHUNK_SIZE, COUNTERFACTUALS_PER_SHARE)
            
            filename = f'counterfactual_results/share_{share_value:.2f}_chunk_{chunk_start}_{chunk_end}.pkl'
            
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    chunk_results = pickle.load(f)
                    share_experiments.append(chunk_results)
                print(f"  Loaded chunk {chunk_id}")
            else:
                print(f"  Warning: {filename} not found")
        
        if share_experiments:
            # Merge chunks for this share value by combining experiment data
            # This is simplified - you might want more sophisticated merging
            all_share_results[share_value] = share_experiments
    
    # Save merged results
    print(f"\\nSaving merged counterfactual results...")
    
    with open('counterfactual_results/all_counterfactuals_merged.pkl', 'wb') as f:
        pickle.dump(all_share_results, f)
    
    print(f"Merged results saved to counterfactual_results/all_counterfactuals_merged.pkl")
    
    # Basic summary
    print(f"\\nSummary:")
    print(f"Total share values processed: {len(all_share_results)}")
    for share, data in all_share_results.items():
        print(f"  Share {share:.2f}: {len(data)} chunks")

def main():
    print("Parallel Counterfactual Analysis")
    print("=" * 50)
    print(f"Share values: {len(SHARES)} values from {SHARES[0]:.2f} to {SHARES[-1]:.2f}")
    print(f"Experiments per share: {COUNTERFACTUALS_PER_SHARE}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Jobs per share: {N_CHUNKS_PER_SHARE}")
    print(f"Total jobs: {len(SHARES) * N_CHUNKS_PER_SHARE}")
    print(f"Cores per job: {CORES_PER_JOB}")
    print(f"Total cores used: {len(SHARES) * N_CHUNKS_PER_SHARE * CORES_PER_JOB}")
    
    response = input(f"\\nSubmit {len(SHARES) * N_CHUNKS_PER_SHARE} jobs? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    # Submit jobs
    job_ids, total_jobs = submit_jobs()
    
    if job_ids:
        print(f"\\nSuccessfully submitted {len(job_ids)} jobs")
        
        monitor_response = input("Monitor jobs? (y/N): ")
        if monitor_response.lower() == 'y':
            monitor_jobs(job_ids)
            
            merge_response = input("Merge results? (y/N): ")
            if merge_response.lower() == 'y':
                merge_results()
    
    print("\\nDone!")

if __name__ == "__main__":
    main()
