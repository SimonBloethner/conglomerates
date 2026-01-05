#!/usr/bin/env python3
"""
Diagnose the actual structure of sensitivity analysis results
without running the full analysis
"""
import pickle
import numpy as np
from pathlib import Path
from compare_parametrizations import ParametrizationComparator

# Load one parametrization
comparator = ParametrizationComparator(results_base_dir='results')
result_file = Path('results/robustness_exponential_baseline/counterfactual_results_final.pkl')

print(f"Loading: {result_file}")
with open(result_file, 'rb') as f:
    data = pickle.load(f)

# Prepare param_data structure
shares = sorted(data.keys())
param_data = {
    'shares': shares,
    'metrics': {}
}

# Extract metrics
for metric_name in ['mean_members_avg', 'num_cong_avg', 'market_share_quantiles_avg', 'gini_quantiles_avg']:
    param_data['metrics'][metric_name] = {}
    for share in shares:
        if metric_name in data[share]:
            param_data['metrics'][metric_name][share] = data[share][metric_name]

print("\nRunning sensitivity analysis on ONE metric to inspect structure...")

# Test with gini_quantiles_avg (quantile-based)
print("\n1. Testing gini_quantiles_avg (quantile metric):")
gini_results, _ = comparator.fit_alpha_sensitivity_models(param_data, 'gini_quantiles_avg')

if gini_results:
    print(f"   Result type: {type(gini_results)}")
    print(f"   Keys: {list(gini_results.keys())}")

    # Check first quantile
    first_quantile = list(gini_results.keys())[0]
    print(f"\n   First quantile '{first_quantile}':")
    print(f"     Keys: {list(gini_results[first_quantile].keys())}")

    if 'temporal' in gini_results[first_quantile]:
        temporal = gini_results[first_quantile]['temporal']
        print(f"     Temporal keys: {list(temporal.keys())}")
        for key, val in list(temporal.items())[:2]:
            print(f"       {key}: type={type(val).__name__}, shape={np.array(val).shape}")

    if 'panel' in gini_results[first_quantile]:
        panel = gini_results[first_quantile]['panel']
        print(f"     Panel keys: {list(panel.keys())}")
        for key, val in list(panel.items())[:3]:
            is_scalar = np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0)
            print(f"       {key}: type={type(val).__name__}, scalar={is_scalar}, value={val}")

# Test with mean_members_avg (scalar metric)
print("\n2. Testing mean_members_avg (scalar metric):")
param_data['metrics']['mean_members_avg'] = {}
for share in shares:
    if 'mean_members_avg' in data[share]:
        param_data['metrics']['mean_members_avg'][share] = data[share]['mean_members_avg']

mean_results, _ = comparator.fit_alpha_sensitivity_models(param_data, 'mean_members_avg')

if mean_results:
    print(f"   Result type: {type(mean_results)}")
    print(f"   Keys: {list(mean_results.keys())}")

    if 'temporal' in mean_results:
        temporal = mean_results['temporal']
        print(f"   Temporal keys: {list(temporal.keys())}")
        for key, val in list(temporal.items())[:2]:
            print(f"     {key}: type={type(val).__name__}, shape={np.array(val).shape}")

    if 'panel' in mean_results:
        panel = mean_results['panel']
        print(f"   Panel keys: {list(panel.keys())}")
        for key, val in list(panel.items())[:3]:
            is_scalar = np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0)
            print(f"     {key}: type={type(val).__name__}, scalar={is_scalar}, value={val}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("This shows the actual data structure returned by sensitivity analysis.")
print("Temporal coefficients should be ARRAYS (shape: (n_timesteps,))")
print("Panel coefficients should be SCALARS (single numbers)")