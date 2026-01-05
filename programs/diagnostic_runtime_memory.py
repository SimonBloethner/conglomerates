#!/usr/bin/env python3
"""
Diagnostic script to measure runtime and memory usage across alpha/cost function combinations.
Tests the hypothesis that different cost functions have dramatically different computational complexity.
"""

import time
import subprocess
import os
import sys
from collaborative_growth import model, get_cost_function_defaults

def log_memory_usage(label):
    """Log current memory usage using ps command"""
    try:
        pid = os.getpid()
        result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], 
                              capture_output=True, text=True)
        memory_kb = int(result.stdout.strip())
        memory_gb = memory_kb / 1024 / 1024
        print(f"MEMORY_LOG {label}: {memory_gb:.2f}GB", flush=True)
        return memory_gb
    except Exception as e:
        print(f"MEMORY_LOG {label}: Unable to measure ({e})", flush=True)
        return None

def run_single_experiment(alpha, cost_function):
    """Run one experiment and measure time/memory"""
    
    print(f"\n{'='*50}")
    print(f"Testing: alpha={alpha:.1f}, cost_function={cost_function}")
    print(f"{'='*50}")
    print("Testing BOTH main process AND worker process memory usage")
    
    # Get cost function defaults
    defaults = get_cost_function_defaults(cost_function)
    
    # Full baseline setup
    hyperparameters = {
        'markets': 100,
        'firms_per_market': 100, 
        'steps': 10000,
        'merge_thresh': 0.05,
        'comparison': 4,
        'break_thresh': 0.85,
        'lookback': 50,
        'proportional': False,
        'cost_type': cost_function,
        'c0': defaults['c0'],
        'c1': defaults['c1'], 
        'c2': defaults['c2']
    }
    
    print("Configuration:")
    print(f"  Markets: {hyperparameters['markets']}")
    print(f"  Firms per market: {hyperparameters['firms_per_market']}")
    print(f"  Total firms: {hyperparameters['markets'] * hyperparameters['firms_per_market']}")
    print(f"  Steps: {hyperparameters['steps']}")
    print(f"  Cost function: {cost_function}")
    print(f"  Alpha: {alpha}")
    print(f"  c0: {hyperparameters['c0']}")
    print(f"  c1: {hyperparameters['c1']}")
    print(f"  c2: {hyperparameters['c2']}")
    
    # Memory measurement before
    memory_start = log_memory_usage("BEFORE_EXPERIMENT")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Prepare parameters for the model function (14-parameter list format)
        total_firms = hyperparameters['markets'] * hyperparameters['firms_per_market']
        
        params = [
            hyperparameters['markets'],           # markets
            hyperparameters['firms_per_market'], # firms_per_market  
            hyperparameters['steps'],            # steps
            alpha,                               # share
            total_firms,                         # total_firms
            hyperparameters['merge_thresh'],     # merge_thresh
            hyperparameters['comparison'],       # comparison
            hyperparameters['break_thresh'],     # break_thresh  
            hyperparameters['proportional'],     # proportional
            hyperparameters['lookback'],         # lookback
            hyperparameters['cost_type'],        # cost_type
            hyperparameters['c0'],               # c0
            hyperparameters['c1'],               # c1
            hyperparameters['c2']                # c2
        ]
        
        print(f"DEBUG PARAMS: markets={params[0]}, firms_per_market={params[1]}, steps={params[2]}")
        print(f"DEBUG PARAMS: share={params[3]}, cost_type={params[10]}, c0={params[11]}, c1={params[12]}, c2={params[13]}")
        
        log_memory_usage("AFTER_PARAMS_SETUP")
        
        # Run simulation with memory tracking
        print("🔍 Starting simulation with memory monitoring...")
        results = model(params)
        
        # Peak memory during simulation 
        memory_peak = log_memory_usage("DURING_SIMULATION")
        
        # End timing
        end_time = time.time()
        runtime = end_time - start_time
        
        # Memory measurement after
        memory_end = log_memory_usage("AFTER_EXPERIMENT")
        
        # Results summary
        print(f"\n🎯 RESULTS:")
        print(f"  ⏱️  Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
        memory_increase = None
        if memory_start and memory_end:
            memory_increase = memory_end - memory_start
            print(f"  💾 Memory start: {memory_start:.2f}GB")
            print(f"  💾 Memory end: {memory_end:.2f}GB") 
            print(f"  💾 Memory increase: {memory_increase:.2f}GB")
        
        # Check if we have valid results (model returns list of 10 elements)
        if results is not None and len(results) == 10:
            mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, gini_coefficient, ranks, avg_ranks, mergers_per_period, hyperparameters = results
            print(f"  📊 Simulation completed successfully")
            print(f"  📊 Gini coefficient shape: {gini_coefficient.shape}")
            print(f"  📊 Total simulation steps: {len(mergers_per_period)}")
            print(f"  📊 Cost function used: {hyperparameters.get('cost_type', 'unknown')}")
        else:
            print(f"  ❌ Simulation failed or incomplete")
            if results is not None:
                print(f"  📊 Results type: {type(results)}, length: {len(results) if hasattr(results, '__len__') else 'N/A'}")
            
        return {
            'alpha': alpha,
            'cost_function': cost_function,
            'runtime_seconds': runtime,
            'runtime_minutes': runtime/60,
            'memory_start_gb': memory_start,
            'memory_end_gb': memory_end,
            'memory_increase_gb': memory_increase if memory_start and memory_end else None,
            'success': results is not None,
            'hyperparameters': hyperparameters
        }
        
    except Exception as e:
        end_time = time.time()
        runtime = end_time - start_time
        memory_end = log_memory_usage("AFTER_ERROR")
        
        print(f"\n❌ EXPERIMENT FAILED:")
        print(f"  Error: {str(e)}")
        print(f"  Runtime before failure: {runtime:.1f} seconds")
        
        return {
            'alpha': alpha,
            'cost_function': cost_function, 
            'runtime_seconds': runtime,
            'runtime_minutes': runtime/60,
            'memory_start_gb': memory_start,
            'memory_end_gb': memory_end,
            'error': str(e),
            'success': False,
            'hyperparameters': hyperparameters
        }

def main():
    """Run diagnostic across all alpha/cost function combinations"""
    
    # Test matrix
    alphas = [0.0, 0.1]
    cost_functions = ['linear', 'quadratic', 'exponential', 'power_law']
    
    print("COMPUTATIONAL DIAGNOSTICS")
    print("=" * 80)
    print("Testing runtime and memory usage across alpha/cost function combinations")
    print(f"Alpha values: {alphas}")
    print(f"Cost functions: {cost_functions}")
    print(f"Total tests: {len(alphas)} × {len(cost_functions)} = {len(alphas) * len(cost_functions)}")
    print("Configuration: 100 markets × 100 firms × 10,000 steps")
    print("=" * 80)
    
    results = []
    
    for alpha in alphas:
        for cost_function in cost_functions:
            result = run_single_experiment(alpha, cost_function)
            results.append(result)
    
    # Summary table
    print(f"\n\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    print(f"{'Alpha':<8} {'Cost Function':<12} {'Runtime (min)':<14} {'Memory (GB)':<12} {'Status'}")
    print(f"{'-'*80}")
    
    for result in results:
        alpha = result['alpha']
        cost_fn = result['cost_function'] 
        runtime = f"{result['runtime_minutes']:.1f}"
        memory = f"{result['memory_increase_gb']:.2f}" if result.get('memory_increase_gb') else "N/A"
        status = "✅ Success" if result['success'] else "❌ Failed"
        
        print(f"{alpha:<8.1f} {cost_fn:<12} {runtime:<14} {memory:<12} {status}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Runtime analysis
        print("\n⏱️ RUNTIME ANALYSIS:")
        alpha_0_results = [r for r in successful_results if r['alpha'] == 0.0]
        alpha_1_results = [r for r in successful_results if r['alpha'] == 0.1]
        
        if alpha_0_results and alpha_1_results:
            avg_runtime_alpha_0 = sum(r['runtime_minutes'] for r in alpha_0_results) / len(alpha_0_results)
            avg_runtime_alpha_1 = sum(r['runtime_minutes'] for r in alpha_1_results) / len(alpha_1_results)
            
            print(f"  Average runtime alpha=0.0: {avg_runtime_alpha_0:.1f} minutes")
            print(f"  Average runtime alpha=0.1: {avg_runtime_alpha_1:.1f} minutes")
            print(f"  Collaboration overhead: {((avg_runtime_alpha_1/avg_runtime_alpha_0)-1)*100:.1f}%")
        
        # Cost function analysis
        print(f"\n🔧 COST FUNCTION ANALYSIS:")
        cost_runtimes = {}
        for cost_fn in cost_functions:
            cost_results = [r for r in successful_results if r['cost_function'] == cost_fn]
            if cost_results:
                avg_runtime = sum(r['runtime_minutes'] for r in cost_results) / len(cost_results)
                cost_runtimes[cost_fn] = avg_runtime
                print(f"  {cost_fn}: {avg_runtime:.1f} minutes average")
        
        # Identify bottlenecks
        if cost_runtimes:
            slowest_cost = max(cost_runtimes.items(), key=lambda x: x[1])
            fastest_cost = min(cost_runtimes.items(), key=lambda x: x[1])
            print(f"\n🏆 Fastest cost function: {fastest_cost[0]} ({fastest_cost[1]:.1f} min)")
            print(f"🐌 Slowest cost function: {slowest_cost[0]} ({slowest_cost[1]:.1f} min)")
            speedup_factor = slowest_cost[1] / fastest_cost[1]
            print(f"📊 Slowest is {speedup_factor:.1f}x slower than fastest")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("❌ Failed experiments detected:")
        for result in failed_results:
            print(f"  - {result['cost_function']} alpha={result['alpha']}: {result.get('error', 'Unknown error')}")
    
    if successful_results:
        max_runtime = max(r['runtime_minutes'] for r in successful_results)
        print(f"⏱️ Maximum runtime observed: {max_runtime:.1f} minutes")
        print(f"💡 Recommended SLURM time limit: {max_runtime * 1.5:.0f} minutes ({max_runtime * 1.5 / 60:.1f} hours)")
        
        if any(r.get('memory_increase_gb', 0) > 5 for r in successful_results):
            print(f"💾 High memory usage detected - consider memory optimizations")
        else:
            print(f"💾 Memory usage appears manageable")

if __name__ == "__main__":
    main()