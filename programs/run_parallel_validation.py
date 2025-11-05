#!/usr/bin/env python3
"""
Script to run hyperparameter validation across multiple SLURM jobs in parallel
"""

import os
import subprocess
import time

# Configuration
TOTAL_PARAMETER_COMBINATIONS = 3840  # 4*4*5*4*3*4
CHUNK_SIZE = 160  # Each job processes 160 parameter combinations (sized for 24h cluster limit)
N_CHUNKS = TOTAL_PARAMETER_COMBINATIONS // CHUNK_SIZE
CORES_PER_JOB = 64
TIME_LIMIT = "23:00:00"  # 23 hours per job (1h buffer under 24h cluster limit)

def create_directories():
    """Create organized directory structure"""
    directories = ['slurm_logs', 'slurm_scripts', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
def create_slurm_script(chunk_id, chunk_start, chunk_end):
    """Create a SLURM batch script for a specific chunk"""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=hyperparam_chunk_{chunk_id}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task={CORES_PER_JOB}
#SBATCH --time={TIME_LIMIT}
#SBATCH --output=slurm_logs/hyperparam_chunk_{chunk_id}_%j.out
#SBATCH --error=slurm_logs/hyperparam_chunk_{chunk_id}_%j.err

# Load required modules
module load python/3.13.3

# Change to the correct directory
cd {os.getcwd()}

# Run the hyperparameter validation for this chunk
echo "Starting chunk {chunk_id}: parameters {chunk_start}-{chunk_end}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"

python3 hyperparameter_validation.py --chunk {chunk_start}_{chunk_end} --n_cores {CORES_PER_JOB}

echo "Chunk {chunk_id} completed at: $(date)"
"""
    
    script_filename = f"slurm_scripts/job_chunk_{chunk_id}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    
    return script_filename

def submit_jobs():
    """Submit all chunk jobs to SLURM"""
    # Create organized directory structure
    create_directories()
    
    job_ids = []
    
    for chunk_id in range(N_CHUNKS):
        chunk_start = chunk_id * CHUNK_SIZE
        chunk_end = min((chunk_id + 1) * CHUNK_SIZE, TOTAL_PARAMETER_COMBINATIONS)
        
        # Create SLURM script
        script_file = create_slurm_script(chunk_id, chunk_start, chunk_end)
        
        # Submit job
        result = subprocess.run(['sbatch', script_file], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f"Submitted chunk {chunk_id} (params {chunk_start}-{chunk_end}): Job ID {job_id}")
        else:
            print(f"Failed to submit chunk {chunk_id}: {result.stderr}")
    
    return job_ids

def monitor_jobs(job_ids):
    """Monitor job progress"""
    print(f"\nMonitoring {len(job_ids)} jobs...")
    print("Use 'squeue -u $USER' to check job status")
    print("Use 'tail -f slurm_logs/hyperparam_chunk_*_*.out' to monitor progress")
    print("Use 'ls slurm_logs/' to see all log files")
    
    # Wait for all jobs to complete
    while True:
        result = subprocess.run(['squeue', '-u', os.getenv('USER'), '--noheader'], 
                              capture_output=True, text=True)
        
        running_jobs = [line.split()[0] for line in result.stdout.strip().split('\n') if line]
        our_running = [jid for jid in job_ids if jid in running_jobs]
        
        if not our_running:
            print("\nAll jobs completed!")
            break
        
        print(f"\rJobs still running: {len(our_running)}/{len(job_ids)}", end='', flush=True)
        time.sleep(30)  # Check every 30 seconds

def merge_results():
    """Merge results from all chunks into a single file"""
    import pickle
    import json
    
    all_results = []
    
    for chunk_id in range(N_CHUNKS):
        chunk_start = chunk_id * CHUNK_SIZE
        chunk_end = min((chunk_id + 1) * CHUNK_SIZE, TOTAL_PARAMETER_COMBINATIONS)
        filename = f'hyperparameter_results_chunk_{chunk_start}_{chunk_end}.pkl'
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                chunk_results = pickle.load(f)
                all_results.extend(chunk_results)
            print(f"Loaded {len(chunk_results)} results from chunk {chunk_id}")
        else:
            print(f"Warning: {filename} not found")
    
    # Save merged results
    print(f"\nMerging {len(all_results)} total results...")
    
    with open('results/hyperparameter_results_merged.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    with open('results/hyperparameter_results_merged.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Merged results saved to results/hyperparameter_results_merged.pkl/json")
    
    # Basic analysis
    if all_results:
        print(f"\nBasic Analysis:")
        print(f"Total parameter sets processed: {len(all_results)}")
        
        best_gini = min(all_results, key=lambda x: x.get('final_gini_mean_mean', float('inf')))
        best_mobility = max(all_results, key=lambda x: x.get('rank_mobility_range_mean', 0))
        
        print(f"Lowest inequality parameters: {best_gini['parameters']}")
        print(f"Highest mobility parameters: {best_mobility['parameters']}")

def main():
    print("Multi-Job Hyperparameter Validation")
    print("=" * 50)
    print(f"Total parameter combinations: {TOTAL_PARAMETER_COMBINATIONS}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Number of jobs: {N_CHUNKS}")
    print(f"Cores per job: {CORES_PER_JOB}")
    print(f"Total cores used: {N_CHUNKS * CORES_PER_JOB}")
    
    response = input(f"\nSubmit {N_CHUNKS} jobs? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    # Submit jobs
    job_ids = submit_jobs()
    
    if job_ids:
        print(f"\nSuccessfully submitted {len(job_ids)} jobs")
        
        monitor_response = input("Monitor jobs? (y/N): ")
        if monitor_response.lower() == 'y':
            monitor_jobs(job_ids)
            
            merge_response = input("Merge results? (y/N): ")
            if merge_response.lower() == 'y':
                merge_results()
    
    print("\nDone!")

if __name__ == "__main__":
    main()