#!/usr/bin/env python3
"""
Test ONLY the aggregation step using pre-computed sensitivity analysis results
This simulates what _aggregate_by_cost_function does without running the full analysis
"""
import pickle
import numpy as np
from pathlib import Path
from compare_parametrizations import ParametrizationComparator

print("Testing aggregation in isolation...\n")

comparator = ParametrizationComparator(results_base_dir='results')

# Manually build the results structure that _aggregate_by_cost_function expects
results = {
    'parametrizations': {},
    'cross_parametrization': {}
}

# Load ALL parametrizations and compute their sensitivity analysis
robustness_dirs = list(Path('results').glob('robustness_*'))  # ALL parametrizations

print(f"Loading {len(robustness_dirs)} parametrizations and computing sensitivity...\n")
print("This may take a few minutes...\n")

for param_dir in robustness_dirs:
    result_file = param_dir / 'counterfactual_results_final.pkl'

    if not result_file.exists():
        continue

    print(f"Processing: {param_dir.name}")

    with open(result_file, 'rb') as f:
        data = pickle.load(f)

    # Get cost type
    first_share = data[0.0]
    hyperparams = first_share.get('hyperparameters', {})
    cost_type = hyperparams.get('cost_type', 'unknown')

    # Prepare param_data
    shares = sorted(data.keys())
    param_data = {
        'shares': shares,
        'metrics': {}
    }

    # Extract metrics
    for metric_name in ['gini_quantiles_avg', 'market_share_quantiles_avg', 'mean_members_avg', 'num_cong_avg']:
        param_data['metrics'][metric_name] = {}
        for share in shares:
            if metric_name in data[share]:
                param_data['metrics'][metric_name][share] = data[share][metric_name]

    # Build param_results structure
    param_results = {
        'cost_type': cost_type,
        'hyperparameters': hyperparams,
        'metrics': {}
    }

    # Compute sensitivity for each metric
    for metric_name in ['gini_quantiles_avg', 'market_share_quantiles_avg', 'mean_members_avg', 'num_cong_avg']:
        sensitivity_results, _ = comparator.fit_alpha_sensitivity_models(param_data, metric_name)
        if sensitivity_results:
            param_results['metrics'][metric_name] = sensitivity_results

    results['parametrizations'][param_dir.name] = param_results

print(f"\n{'='*80}")
print(f"Running aggregation on {len(results['parametrizations'])} parametrizations...")
print(f"{'='*80}\n")

# Now run the aggregation with debugging
cost_aggregates = {}

for param_name, param_results in results['parametrizations'].items():
    cost_type = param_results['cost_type']

    if cost_type not in cost_aggregates:
        cost_aggregates[cost_type] = {
            'parametrizations': [],
            'temporal_coefficients': {},
            'panel_coefficients': {}
        }

    cost_aggregates[cost_type]['parametrizations'].append(param_name)

    # Collect temporal and panel coefficients for aggregation
    for metric_name, metric_data in param_results.get('metrics', {}).items():
        if isinstance(metric_data, dict):
            # Handle quantile-based metrics
            if any(q in metric_data for q in ['10th', '25th', '50th', '75th', '90th', '99th', '100th']):
                for quantile_name, quantile_data in metric_data.items():
                    if isinstance(quantile_data, dict):
                        # Handle temporal coefficients
                        if 'temporal' in quantile_data:
                            temporal_data = quantile_data['temporal']
                            for coeff_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                                if coeff_name in temporal_data:
                                    key = f"{metric_name}_{quantile_name}_temporal_{coeff_name}"
                                    if key not in cost_aggregates[cost_type]['temporal_coefficients']:
                                        cost_aggregates[cost_type]['temporal_coefficients'][key] = []
                                    cost_aggregates[cost_type]['temporal_coefficients'][key].append(
                                        temporal_data[coeff_name]
                                    )

                        # Handle panel coefficients
                        if 'panel' in quantile_data:
                            panel_data = quantile_data['panel']
                            for coeff_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                                if coeff_name in panel_data:
                                    key = f"{metric_name}_{quantile_name}_panel_{coeff_name}"
                                    if key not in cost_aggregates[cost_type]['panel_coefficients']:
                                        cost_aggregates[cost_type]['panel_coefficients'][key] = []
                                    cost_aggregates[cost_type]['panel_coefficients'][key].append(
                                        panel_data[coeff_name]
                                    )
            else:
                # Handle scalar metrics with temporal/panel structure
                if 'temporal' in metric_data:
                    temporal_data = metric_data['temporal']
                    for coeff_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                        if coeff_name in temporal_data:
                            key = f"{metric_name}_temporal_{coeff_name}"
                            if key not in cost_aggregates[cost_type]['temporal_coefficients']:
                                cost_aggregates[cost_type]['temporal_coefficients'][key] = []
                            cost_aggregates[cost_type]['temporal_coefficients'][key].append(
                                temporal_data[coeff_name]
                            )

                if 'panel' in metric_data:
                    panel_data = metric_data['panel']
                    for coeff_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                        if coeff_name in panel_data:
                            key = f"{metric_name}_panel_{coeff_name}"
                            if key not in cost_aggregates[cost_type]['panel_coefficients']:
                                cost_aggregates[cost_type]['panel_coefficients'][key] = []
                            cost_aggregates[cost_type]['panel_coefficients'][key].append(
                                panel_data[coeff_name]
                            )

print(f"Aggregation collection complete. Now attempting to stack arrays...\n")

# Now try to stack with debugging
for cost_type, cost_data in cost_aggregates.items():
    print(f"Processing {cost_type} ({len(cost_data['parametrizations'])} parametrizations):")

    cost_data['temporal_stats'] = {}
    cost_data['panel_stats'] = {}

    # Process temporal coefficients (arrays) - THIS IS WHERE THE ERROR HAPPENS
    print(f"  Temporal coefficients to process: {len(cost_data['temporal_coefficients'])}")

    for key, coefficient_arrays in cost_data['temporal_coefficients'].items():
        if coefficient_arrays:
            # DEBUG: Check shapes before stacking
            shapes = [np.array(arr).shape for arr in coefficient_arrays]
            unique_shapes = set(shapes)

            if len(unique_shapes) > 1:
                print(f"\n  ❌ ERROR in key: {key}")
                print(f"     Number of arrays: {len(coefficient_arrays)}")
                print(f"     Unique shapes: {unique_shapes}")
                print(f"     First 5 arrays:")
                for i in range(min(5, len(coefficient_arrays))):
                    arr = coefficient_arrays[i]
                    print(f"       [{i}] shape={np.array(arr).shape}, type={type(arr).__name__}, scalar={np.isscalar(arr)}")
                    if np.isscalar(arr):
                        print(f"            value={arr}")
                print(f"     Skipping this key...\n")
                continue

            # Try to stack
            try:
                stacked = np.array(coefficient_arrays)
                print(f"  ✅ {key}: stacked shape {stacked.shape}")
            except ValueError as e:
                print(f"  ❌ {key}: FAILED to stack - {e}")

    # Process panel coefficients (scalars)
    print(f"\n  Panel coefficients to process: {len(cost_data['panel_coefficients'])}")

    for key, coefficient_scalars in cost_data['panel_coefficients'].items():
        if coefficient_scalars:
            are_scalars = all(np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0)
                              for val in coefficient_scalars)
            if not are_scalars:
                print(f"  ❌ {key}: contains non-scalars!")
            else:
                scalars = np.array(coefficient_scalars)
                print(f"  ✅ {key}: {len(scalars)} scalars")

print(f"\n{'='*80}")
print("Aggregation test complete!")
print(f"{'='*80}")