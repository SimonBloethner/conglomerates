#!/usr/bin/env python3
"""
§5 Diagnostic Run

Run small-scale diagnostics to verify model behavior with all fixes.
- M=N=50 (markets=firms_per_market=50)
- T=2000 steps
- α ∈ {0, 0.02, 0.1, 0.3, 0.5}
- All 4 cost types
- 5 replications
"""

import os
import pickle
import numpy as np
from datetime import datetime
from collaborative_growth import model, seed_numba, get_cost_function_defaults

# Configuration
MARKETS = 50
FIRMS_PER_MARKET = 50
STEPS = 2000
ALPHA_VALUES = [0.0, 0.02, 0.1, 0.3, 0.5]
COST_TYPES = ['linear', 'quadratic', 'exponential', 'power_law']
N_REPLICATIONS = 5
OUTPUT_DIR = 'diagnostics'

def run_single_experiment(params, seed, market_corr='identity'):
    """Run a single model experiment and return key metrics."""
    seed_numba(seed)
    np.random.seed(seed)
    
    result = model(params, seed=seed, market_corr=market_corr)
    
    # Unpack results
    gini_ts = result[0]
    active_ts = result[1]
    wealth_ts = result[2]
    conglomerate_sizes_ts = result[3]
    market_shares = result[4]
    proposals_per_period = result[5] if len(result) > 5 else None
    
    return {
        'gini_ts': gini_ts,
        'active_ts': active_ts,
        'wealth_ts': wealth_ts,
        'conglomerate_sizes_ts': conglomerate_sizes_ts,
        'market_shares': market_shares,
        'proposals_per_period': proposals_per_period,
    }


def run_diagnostics():
    """Run all diagnostic experiments."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {}
    total_runs = len(COST_TYPES) * len(ALPHA_VALUES) * N_REPLICATIONS
    run_count = 0
    
    print(f"Starting diagnostic run: {total_runs} experiments")
    print(f"Config: M={MARKETS}, N={FIRMS_PER_MARKET}, T={STEPS}")
    print(f"α values: {ALPHA_VALUES}")
    print(f"Cost types: {COST_TYPES}")
    print(f"Replications: {N_REPLICATIONS}")
    print()
    
    total_firms = MARKETS * FIRMS_PER_MARKET
    
    for cost_type in COST_TYPES:
        defaults = get_cost_function_defaults(cost_type)
        results[cost_type] = {}
        
        for alpha in ALPHA_VALUES:
            results[cost_type][alpha] = []
            
            for rep in range(N_REPLICATIONS):
                run_count += 1
                seed = hash((cost_type, alpha, rep)) % (2**32)
                
                # Build params list for model
                # Correct order: markets, firms_per_market, steps, share, total_firms, 
                #                merge_thresh, comparison, break_thresh, proportional, lookback,
                #                cost_type, c0, c1, c2
                params = [
                    MARKETS,             # markets
                    FIRMS_PER_MARKET,    # firms_per_market
                    STEPS,               # steps
                    alpha,               # share (α)
                    total_firms,         # total_firms (M × N)
                    0.05,                # merge_thresh
                    4,                   # comparison (n_bins for histogram)
                    0.85,                # break_thresh
                    False,               # proportional
                    50,                  # lookback
                    cost_type,           # cost_type
                    defaults['c0'],      # c0
                    defaults['c1'],      # c1
                    defaults['c2'],      # c2
                ]
                
                print(f"[{run_count}/{total_runs}] {cost_type}, α={alpha}, rep={rep+1}... ", end='', flush=True)
                
                try:
                    exp_result = run_single_experiment(params, seed)
                    results[cost_type][alpha].append(exp_result)
                    
                    # Print quick summary
                    final_gini = exp_result['gini_ts'][-1] if len(exp_result['gini_ts']) > 0 else np.nan
                    final_active = exp_result['active_ts'][-1] if len(exp_result['active_ts']) > 0 else np.nan
                    total_proposals = exp_result['proposals_per_period'].sum() if exp_result['proposals_per_period'] is not None else 0
                    print(f"Gini={final_gini:.4f}, Active={final_active:.0f}, Proposals={total_proposals}")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    results[cost_type][alpha].append({'error': str(e)})
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f'diagnostic_results_{timestamp}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_file}")
    
    # Generate summary
    generate_summary(results, OUTPUT_DIR, timestamp)
    
    return results


def generate_summary(results, output_dir, timestamp):
    """Generate summary statistics and markdown report."""
    summary_file = os.path.join(output_dir, f'diagnostic_summary_{timestamp}.md')
    
    with open(summary_file, 'w') as f:
        f.write("# Diagnostic Run Summary\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Markets: {MARKETS}\n")
        f.write(f"- Firms per market: {FIRMS_PER_MARKET}\n")
        f.write(f"- Steps: {STEPS}\n")
        f.write(f"- Replications: {N_REPLICATIONS}\n")
        f.write(f"- α values: {ALPHA_VALUES}\n\n")
        
        f.write("## Final Gini Coefficient by Cost Type and α\n\n")
        f.write("| Cost Type | α=0.00 | α=0.02 | α=0.10 | α=0.30 | α=0.50 |\n")
        f.write("|-----------|--------|--------|--------|--------|--------|\n")
        
        for cost_type in COST_TYPES:
            row = f"| {cost_type} |"
            for alpha in ALPHA_VALUES:
                ginis = []
                for rep_result in results[cost_type][alpha]:
                    if 'error' not in rep_result:
                        ginis.append(rep_result['gini_ts'][-1])
                if ginis:
                    mean_gini = np.mean(ginis)
                    std_gini = np.std(ginis)
                    row += f" {mean_gini:.3f}±{std_gini:.3f} |"
                else:
                    row += " ERROR |"
            f.write(row + "\n")
        
        f.write("\n## Merger Activity (Total Proposals) by Cost Type and α\n\n")
        f.write("| Cost Type | α=0.00 | α=0.02 | α=0.10 | α=0.30 | α=0.50 |\n")
        f.write("|-----------|--------|--------|--------|--------|--------|\n")
        
        for cost_type in COST_TYPES:
            row = f"| {cost_type} |"
            for alpha in ALPHA_VALUES:
                proposals = []
                for rep_result in results[cost_type][alpha]:
                    if 'error' not in rep_result and rep_result['proposals_per_period'] is not None:
                        proposals.append(rep_result['proposals_per_period'].sum())
                if proposals:
                    mean_prop = np.mean(proposals)
                    row += f" {mean_prop:.0f} |"
                else:
                    row += " - |"
            f.write(row + "\n")
        
        f.write("\n## Active Firms at End by Cost Type and α\n\n")
        f.write("| Cost Type | α=0.00 | α=0.02 | α=0.10 | α=0.30 | α=0.50 |\n")
        f.write("|-----------|--------|--------|--------|--------|--------|\n")
        
        for cost_type in COST_TYPES:
            row = f"| {cost_type} |"
            for alpha in ALPHA_VALUES:
                actives = []
                for rep_result in results[cost_type][alpha]:
                    if 'error' not in rep_result:
                        actives.append(rep_result['active_ts'][-1])
                if actives:
                    mean_active = np.mean(actives)
                    row += f" {mean_active:.0f} |"
                else:
                    row += " ERROR |"
            f.write(row + "\n")
    
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    run_diagnostics()
