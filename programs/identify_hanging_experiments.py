#!/usr/bin/env python3
"""
Identify which specific experiments are hanging by analyzing interleaved SLURM logs
"""

import re
import glob
import sys
from collections import defaultdict

def analyze_hanging_experiments(log_file):
    """Parse a single log file to identify completed vs hanging experiments"""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract all experiment starts
    start_pattern = r'Starting experiment (\d+) for share ([0-9.]+)'
    starts = re.findall(start_pattern, content)
    
    # Extract all experiment completions
    completion_pattern = r'Completed: share=([0-9.]+), exp=(\d+), runtime=([0-9.]+)s'
    completions = re.findall(completion_pattern, content)
    
    # Convert to sets for easy comparison
    started_experiments = {(float(share), int(exp)) for exp, share in starts}
    completed_experiments = {(float(share), int(exp)) for share, exp, runtime in completions}
    
    # Find hanging experiments
    hanging = started_experiments - completed_experiments
    
    print(f"\n=== Analysis for {log_file} ===")
    print(f"Experiments started: {len(started_experiments)}")
    print(f"Experiments completed: {len(completed_experiments)}")
    print(f"Experiments hanging: {len(hanging)}")
    
    if hanging:
        print("Hanging experiments:")
        for share, exp in sorted(hanging):
            print(f"  Share {share:.2f}, exp {exp}")
            
            # Try to find the last debug message for this specific experiment
            # This is tricky with interleaved logs, but we can try to find context
            exp_context = find_experiment_context(content, share, exp)
            if exp_context:
                print(f"    Last context: {exp_context}")
    
    return hanging, completed_experiments

def find_experiment_context(content, target_share, target_exp):
    """Try to find the last debug context for a specific experiment"""
    
    # Look for the experiment start
    start_pattern = f'Starting experiment {target_exp} for share {target_share:.2f}'
    
    lines = content.split('\n')
    experiment_started = False
    last_debug = None
    
    for i, line in enumerate(lines):
        if start_pattern in line:
            experiment_started = True
            continue
            
        # If this experiment completed, stop looking
        if experiment_started and f'Completed: share={target_share:.2f}, exp={target_exp}' in line:
            return "COMPLETED"
            
        # Look for debug messages after the experiment started
        if experiment_started and 'DEBUG:' in line:
            last_debug = line.strip()
            
        # If we see another experiment start, this gets complicated due to interleaving
        if experiment_started and 'Starting experiment' in line and start_pattern not in line:
            # We've moved to another experiment, return what we found
            break
    
    return last_debug

def main():
    # Find all log files
    log_pattern = 'counterfactual_slurm_logs/*.out'
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"No log files found matching {log_pattern}")
        print("Make sure you're running this from the correct directory")
        return
    
    print(f"Analyzing {len(log_files)} log files...")
    
    total_hanging = 0
    total_completed = 0
    hanging_by_share = defaultdict(list)
    
    for log_file in sorted(log_files):
        hanging, completed = analyze_hanging_experiments(log_file)
        total_hanging += len(hanging)
        total_completed += len(completed)
        
        # Group hanging experiments by share value
        for share, exp in hanging:
            hanging_by_share[share].append((log_file, exp))
    
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total completed experiments: {total_completed}")
    print(f"Total hanging experiments: {total_hanging}")
    
    if hanging_by_share:
        print(f"\nHanging experiments by share value:")
        for share in sorted(hanging_by_share.keys()):
            experiments = hanging_by_share[share]
            print(f"  Share {share:.2f}: {len(experiments)} hanging")
            for log_file, exp in experiments[:3]:  # Show first 3 examples
                print(f"    {log_file}: exp {exp}")
            if len(experiments) > 3:
                print(f"    ... and {len(experiments) - 3} more")

if __name__ == "__main__":
    main()