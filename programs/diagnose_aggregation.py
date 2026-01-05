#!/usr/bin/env python3
"""
Simulate the aggregation process to find where inhomogeneous shapes occur
"""
import pickle
import numpy as np
from pathlib import Path
from compare_parametrizations import ParametrizationComparator

print("Simulating the aggregation process...\n")

comparator = ParametrizationComparator(results_base_dir='results')

# Load parametrizations
robustness_dirs = list(Path('results').glob('robustness_*'))

# Group by cost type
cost_groups = {}
for param_dir in sorted(robustness_dirs)[:20]:  # First 20 for speed
    result_file = param_dir / 'counterfactual_results_final.pkl'

    if result_file.exists():
        with open(result_file, 'rb') as f:
            data = pickle.load(f)

        # Get cost type from hyperparameters
        first_share = data[0.0]
        hyperparams = first_share.get('hyperparameters', {})
        cost_type = hyperparams.get('cost_type', 'unknown')

        if cost_type not in cost_groups:
            cost_groups[cost_type] = []

        cost_groups[cost_type].append({
            'name': param_dir.name,
            'data': data
        })

print(f"Found {sum(len(v) for v in cost_groups.values())} parametrizations")
for cost_type, params in cost_groups.items():
    print(f"  {cost_type}: {len(params)} parametrizations")

# Now simulate aggregation for one cost type
print("\nSimulating aggregation for first cost type...")
cost_type = list(cost_groups.keys())[0]
params = cost_groups[cost_type]

print(f"\nProcessing {cost_type} with {len(params)} parametrizations\n")

# Simulate what _aggregate_by_cost_function does
temporal_coefficients = {}
panel_coefficients = {}

for param_info in params:
    param_name = param_info['name']
    data = param_info['data']

    # Prepare param_data
    shares = sorted(data.keys())
    param_data = {
        'shares': shares,
        'metrics': {}
    }

    # Extract one metric
    metric_name = 'gini_quantiles_avg'
    param_data['metrics'][metric_name] = {}
    for share in shares:
        if metric_name in data[share]:
            param_data['metrics'][metric_name][share] = data[share][metric_name]

    # Run sensitivity analysis
    results, _ = comparator.fit_alpha_sensitivity_models(param_data, metric_name)

    if results:
        # Process quantiles (simulating lines 1063-1087)
        for quantile_name, quantile_data in results.items():
            if isinstance(quantile_data, dict):
                # Temporal coefficients
                if 'temporal' in quantile_data:
                    temporal_data = quantile_data['temporal']
                    for coeff_name in ['linear_beta1', 'quadratic_beta1']:
                        if coeff_name in temporal_data:
                            key = f"{metric_name}_{quantile_name}_temporal_{coeff_name}"
                            if key not in temporal_coefficients:
                                temporal_coefficients[key] = []

                            value = temporal_data[coeff_name]
                            temporal_coefficients[key].append(value)

                            # DEBUG
                            print(f"  {param_name[:30]:<30} | {key[:40]:<40} | shape={np.array(value).shape}")

                # Panel coefficients
                if 'panel' in quantile_data:
                    panel_data = quantile_data['panel']
                    for coeff_name in ['linear_beta1', 'quadratic_beta1']:
                        if coeff_name in panel_data:
                            key = f"{metric_name}_{quantile_name}_panel_{coeff_name}"
                            if key not in panel_coefficients:
                                panel_coefficients[key] = []

                            value = panel_data[coeff_name]
                            panel_coefficients[key].append(value)

print(f"\n{'='*80}")
print("CHECKING FOR INHOMOGENEOUS SHAPES IN TEMPORAL COEFFICIENTS")
print(f"{'='*80}\n")

for key, coefficient_arrays in temporal_coefficients.items():
    shapes = [np.array(arr).shape for arr in coefficient_arrays]
    unique_shapes = set(shapes)

    if len(unique_shapes) > 1:
        print(f"❌ {key}")
        print(f"   Found {len(unique_shapes)} different shapes: {unique_shapes}")
        print(f"   Shape distribution:")
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"     {shape}: {count} arrays")
    else:
        print(f"✅ {key}: all {len(coefficient_arrays)} arrays have shape {list(unique_shapes)[0]}")

print(f"\n{'='*80}")
print("CHECKING PANEL COEFFICIENTS (should all be scalars)")
print(f"{'='*80}\n")

for key, coefficient_scalars in panel_coefficients.items():
    are_scalars = all(np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0)
                      for val in coefficient_scalars)
    print(f"{'✅' if are_scalars else '❌'} {key}: all scalars = {are_scalars}")
    if not are_scalars:
        print(f"   Types: {[type(val).__name__ for val in coefficient_scalars]}")