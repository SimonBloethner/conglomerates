#!/usr/bin/env python3
"""
Check for missing experiments by comparing expected vs actual counts
"""

import re
import glob
import numpy as np
from collections import defaultdict

def analyze_experiment_counts():
    """Analyze expected vs actual experiment counts"""
    
    # Expected configuration from run_parallel_counterfactuals.py
    SHARES = np.arange(0, 0.52, 0.02)  # 26 share values
    COUNTERFACTUALS_PER_SHARE = 50
    
    print(f"Expected configuration:")
    print(f"Share values: {len(SHARES)} (from {SHARES[0]:.2f} to {SHARES[-1]:.2f})")
    print(f"Experiments per share: {COUNTERFACTUALS_PER_SHARE}")
    print(f"Total expected: {len(SHARES) * COUNTERFACTUALS_PER_SHARE}")
    
    # Find all log files
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    print(f"\nAnalyzing {len(log_files)} log files...")
    
    # Count completions by share
    completions_by_share = defaultdict(list)
    total_completions = 0
    
    completion_pattern = r'Completed: share=([0-9.]+), exp=(\d+), runtime=([0-9.]+)s'
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
        
        completions = re.findall(completion_pattern, content)
        for share_str, exp_str, runtime_str in completions:
            share = float(share_str)
            exp = int(exp_str)
            runtime = float(runtime_str)
            completions_by_share[share].append((exp, runtime))
            total_completions += 1
    
    print(f"\n=== ACTUAL RESULTS ===")
    print(f"Total completed experiments: {total_completions}")
    
    missing_total = 0
    
    for share in sorted(SHARES):
        expected = COUNTERFACTUALS_PER_SHARE
        actual = len(completions_by_share[share])
        missing = expected - actual
        missing_total += missing
        
        status = "✅" if missing == 0 else f"❌ missing {missing}"
        print(f"Share {share:.2f}: {actual}/{expected} {status}")
        
        if missing > 0 and actual > 0:
            # Show which experiment numbers completed
            completed_exps = sorted([exp for exp, runtime in completions_by_share[share]])
            expected_exps = list(range(COUNTERFACTUALS_PER_SHARE))
            missing_exps = [exp for exp in expected_exps if exp not in completed_exps]
            print(f"  Missing experiments: {missing_exps}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Expected total: {len(SHARES) * COUNTERFACTUALS_PER_SHARE}")
    print(f"Actual total: {total_completions}")
    print(f"Missing total: {missing_total}")
    
    # Check if any jobs are still running
    print(f"\n=== JOB STATUS CHECK ===")
    print("Run 'squeue -u $USER' to check if jobs are still running")
    
    # Check for any error patterns
    print(f"\n=== ERROR ANALYSIS ===")
    error_pattern = r'ERROR:|TIMEOUT:|Failed|Exception|Traceback'
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
        
        if re.search(error_pattern, content, re.IGNORECASE):
            print(f"Found potential errors in {log_file}")
            # Show last few lines that might contain errors
            lines = content.split('\n')
            for line in lines[-20:]:
                if re.search(error_pattern, line, re.IGNORECASE):
                    print(f"  {line.strip()}")

if __name__ == "__main__":
    analyze_experiment_counts()