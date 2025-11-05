#!/usr/bin/env python3
"""
Debug script to investigate the timeout case: share=0.46, exp=9, seed=13600
"""

import numpy as np
import time
import collaborative_growth

def debug_problematic_case():
    """Reproduce the exact problematic case with detailed logging"""
    
    # Exact parameters from the timeout case
    share = 0.46
    experiment_id = 9
    seed = 13600
    
    print(f"=== DEBUGGING TIMEOUT CASE ===")
    print(f"Share: {share}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Seed: {seed}")
    print(f"Starting at: {time.ctime()}")
    
    # Set the exact same random seed
    np.random.seed(seed)
    
    # Same parameters as in parallel_counterfactuals.py
    markets = 100
    firms_per_market = 100
    steps = 10000
    total_firms = markets * firms_per_market
    merge_thresh = 0.05
    comparison = 4
    break_thresh = 0.85
    proportional = False
    lookback = 50
    b0 = 0.00001
    b1 = 1.2
    
    params = [markets, firms_per_market, steps, share, 
             total_firms, merge_thresh, comparison, 
             break_thresh, proportional, lookback, 
             b0, b1]
    
    print(f"Parameters: {params}")
    
    # Add progress tracking to see where it gets stuck
    start_time = time.time()
    
    try:
        # Run with timeout for safety
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Debug timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout for debugging
        
        print("Starting collaborative_growth.model()...")
        
        # Let's test different parts to isolate the issue
        print("Testing parameter setup...")
        
        # Test just the initialization part
        markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback, b0, b1 = params
        
        print("Testing random number generation...")
        min_mu, max_mu = 0.01, 0.1
        min_sig, max_sig = 0.01, 0.05
        min_bound = np.array([min_mu, min_sig])
        max_bound = np.array([max_mu, max_sig])
        mu_sig_corr = 0.7
        means = [0.1, 0.05]
        
        print("Testing covariance matrix...")
        cov_mat = np.ones((2, 2)) * mu_sig_corr + np.diag(np.tile(1 - mu_sig_corr, 2))
        
        print("Testing multivariate normal generation...")
        growth_vars = np.random.multivariate_normal(means, cov_mat, markets)
        
        print("Testing correlation matrix generation...")
        from scipy.stats import random_correlation
        eigen_vals = np.random.uniform(0.1, 3, markets)
        eigen_vals = eigen_vals * markets / eigen_vals.sum()
        
        print("This might be the problem - random_correlation.rvs...")
        market_corr = random_correlation.rvs(tuple(eigen_vals), random_state=np.random.default_rng())
        
        print("✅ Made it past random_correlation.rvs")
        
        result = collaborative_growth.model(params=params)
        print("✅ Model completed successfully!")
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ Timed out after {elapsed:.2f} seconds")
        print("This confirms the problematic case")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    debug_problematic_case()