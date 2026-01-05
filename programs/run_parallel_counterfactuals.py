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
CHUNK_SIZE = 10  # Each job processes 10 experiments (optimal for multiprocessing overhead)
N_CHUNKS_PER_SHARE = COUNTERFACTUALS_PER_SHARE // CHUNK_SIZE  # 5 chunks per share
CORES_PER_JOB = 10  # 10 cores = 10 experiments (perfect 1:1, avoids pool overhead)
TIME_LIMIT = "0:15:00"  # 15 min per job (validated: 10 experiments in 12 min)

def create_directories():
    """Create organized directory structure"""
    directories = ['counterfactual_slurm_logs', 'counterfactual_slurm_scripts', 'counterfactual_results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_slurm_script(share_value, chunk_id, chunk_start, chunk_end, hyperparams):
    """Create a SLURM batch script for a specific share/chunk combination"""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=s{share_value:.2f}_c{chunk_id}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task={CORES_PER_JOB}
#SBATCH --mem=4G
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
echo "Starting share {share_value:.2f}, chunk {chunk_id}: experiments {chunk_start}-{chunk_end-1}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "Hyperparameters: {hyperparams}"

$VIRTUAL_ENV/bin/python parallel_counterfactuals.py --share {share_value:.2f} --chunk {chunk_start}_{chunk_end} --n_cores {CORES_PER_JOB} --counterfactuals {COUNTERFACTUALS_PER_SHARE} {hyperparams}

echo "Share {share_value:.2f}, chunk {chunk_id} completed at: $(date)"
"""
    
    script_filename = f"counterfactual_slurm_scripts/job_share_{share_value:.2f}_chunk_{chunk_id}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    
    return script_filename

def submit_jobs(args):
    """Submit all counterfactual jobs to SLURM"""
    create_directories()
    
    # Get cost function specific defaults if parameters not specified
    from collaborative_growth import get_cost_function_defaults
    defaults = get_cost_function_defaults(args.cost_type)
    
    c0 = args.c0 if args.c0 is not None else defaults['c0']
    c1 = args.c1 if args.c1 is not None else defaults['c1'] 
    c2 = args.c2 if args.c2 is not None else defaults['c2']
    
    # Build hyperparameter string for command line
    hyperparam_str = f"--markets {args.markets} --firms_per_market {args.firms_per_market} --steps {args.steps} --merge_thresh {args.merge_thresh} --comparison {args.comparison} --break_thresh {args.break_thresh} --lookback {args.lookback} --cost_type {args.cost_type} --c0 {c0} --c1 {c1} --c2 {c2}"
    if args.proportional:
        hyperparam_str += " --proportional"
    if args.scenario_name:
        hyperparam_str += f" --scenario_name {args.scenario_name}"
    
    job_ids = []
    total_jobs = 0
    
    for share_value in SHARES:
        for chunk_id in range(N_CHUNKS_PER_SHARE):
            chunk_start = chunk_id * CHUNK_SIZE
            chunk_end = min((chunk_id + 1) * CHUNK_SIZE, COUNTERFACTUALS_PER_SHARE)
            
            # Create SLURM script for this chunk
            script_file = create_slurm_script(share_value, chunk_id, chunk_start, chunk_end, hyperparam_str)
            
            # Submit job
            result = subprocess.run(['sbatch', script_file], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"Submitted share {share_value:.2f}, chunk {chunk_id} (exp {chunk_start}-{chunk_end-1}): Job ID {job_id}")
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
                print(f"  ✅ Loaded chunk {chunk_id}")
            else:
                print(f"  ❌ Warning: {filename} not found")
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parallel counterfactual analysis on SLURM')
    
    # Basic execution parameters
    parser.add_argument('--counterfactuals', type=int, default=50,
                       help='Number of experiments per share value (default: 50)')
    parser.add_argument('--cores_per_job', type=int, default=3,
                       help='CPU cores per SLURM job (default: 3)')
    parser.add_argument('--time_limit', type=str, default="4:00:00",
                       help='Time limit per job (default: 4:00:00)')
    
    # Hyperparameter arguments
    parser.add_argument('--markets', type=int, default=100,
                       help='Number of markets (default: 100)')
    parser.add_argument('--firms_per_market', type=int, default=100,
                       help='Number of firms per market (default: 100)')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of simulation steps (default: 10000)')
    parser.add_argument('--merge_thresh', type=float, default=0.05,
                       help='Merger threshold (default: 0.05)')
    parser.add_argument('--comparison', type=int, default=4,
                       help='Comparison parameter (default: 4)')
    parser.add_argument('--break_thresh', type=float, default=0.85,
                       help='Breakup threshold (default: 0.85)')
    parser.add_argument('--lookback', type=int, default=50,
                       help='Lookback period for exit decisions (default: 50)')
    parser.add_argument('--proportional', action='store_true',
                       help='Use proportional sharing (default: False)')
    parser.add_argument('--cost_type', type=str, default='power_law',
                       choices=['linear', 'quadratic', 'exponential', 'power_law'],
                       help='Management cost function type (default: power_law)')
    parser.add_argument('--c0', type=float, default=None,
                       help='Base cost parameter c0 (default: cost-function-specific)')
    parser.add_argument('--c1', type=float, default=None,
                       help='Scaling cost parameter c1 (default: cost-function-specific)')
    parser.add_argument('--c2', type=float, default=None,
                       help='Quadratic cost parameter c2 (default: cost-function-specific)')
    # Backward compatibility
    parser.add_argument('--b0', type=float, default=0.00001,
                       help='DEPRECATED: Use --c0 instead. Power law parameter b0 (default: 0.00001)')
    parser.add_argument('--b1', type=float, default=1.2,
                       help='DEPRECATED: Use --c1 instead. Power law parameter b1 (default: 1.2)')
    
    # Robustness analysis scenario naming
    parser.add_argument('--scenario_name', type=str, default=None,
                       help='Name for robustness analysis scenario (organizes results by scenario)')
    
    # Non-interactive mode for SLURM
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip interactive confirmation (for automated/SLURM execution)')
    
    args = parser.parse_args()
    
    # Update global configuration with command line arguments
    global COUNTERFACTUALS_PER_SHARE, CORES_PER_JOB, TIME_LIMIT
    COUNTERFACTUALS_PER_SHARE = args.counterfactuals
    CORES_PER_JOB = args.cores_per_job
    TIME_LIMIT = args.time_limit
    
    print("Parallel Counterfactual Analysis - Chunked Jobs")
    print("=" * 60)
    print(f"Share values: {len(SHARES)} values from {SHARES[0]:.2f} to {SHARES[-1]:.2f}")
    print(f"Experiments per share: {COUNTERFACTUALS_PER_SHARE}")
    print(f"Chunk size: {CHUNK_SIZE} experiments per job")
    print(f"Jobs per share: {N_CHUNKS_PER_SHARE}")
    print(f"Total jobs: {len(SHARES) * N_CHUNKS_PER_SHARE}")
    print(f"Cores per job: {CORES_PER_JOB}")
    print(f"Memory per job: 35GB (measured usage: ~{CORES_PER_JOB * 0.22:.1f}GB = {CORES_PER_JOB} workers × 220MB)")
    
    if not args.yes:
        response = input(f"\\nSubmit {len(SHARES) * N_CHUNKS_PER_SHARE} jobs? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    else:
        print(f"\\nAuto-submitting {len(SHARES) * N_CHUNKS_PER_SHARE} jobs (--yes flag provided)")
    
    # Get cost function specific defaults if parameters not specified  
    from collaborative_growth import get_cost_function_defaults
    defaults = get_cost_function_defaults(args.cost_type)
    
    c0 = args.c0 if args.c0 is not None else defaults['c0']
    c1 = args.c1 if args.c1 is not None else defaults['c1'] 
    c2 = args.c2 if args.c2 is not None else defaults['c2']
    
    # Print configuration
    print("Configuration:")
    print(f"  Markets: {args.markets}")
    print(f"  Firms per market: {args.firms_per_market}")
    print(f"  Total firms: {args.markets * args.firms_per_market}")
    print(f"  Steps: {args.steps}")
    print(f"  Merger threshold: {args.merge_thresh}")
    print(f"  Comparison: {args.comparison}")
    print(f"  Break threshold: {args.break_thresh}")
    print(f"  Lookback: {args.lookback}")
    print(f"  Proportional: {args.proportional}")
    print(f"  Cost type: {args.cost_type}")
    print(f"  c0: {c0} {'(default)' if args.c0 is None else '(specified)'}")
    print(f"  c1: {c1} {'(default)' if args.c1 is None else '(specified)'}")
    print(f"  c2: {c2} {'(default)' if args.c2 is None else '(specified)'}")
    print(f"  Cores per job: {CORES_PER_JOB}")
    print(f"  Counterfactuals per share: {COUNTERFACTUALS_PER_SHARE}")
    print()
    
    # Submit jobs
    job_ids, total_jobs = submit_jobs(args)
    
    if job_ids:
        print(f"\\nSuccessfully submitted {len(job_ids)} jobs")
        
        if not args.yes:
            monitor_response = input("Monitor jobs? (y/N): ")
            if monitor_response.lower() == 'y':
                monitor_jobs(job_ids)
                
                merge_response = input("Merge results? (y/N): ")
                if merge_response.lower() == 'y':
                    merge_results()
        else:
            print("Jobs submitted. Use 'squeue -u $USER' to monitor progress.")
    
    print("\\nDone!")

if __name__ == "__main__":
    main()
