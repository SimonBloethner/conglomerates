#!/usr/bin/env python3
"""
Check ALL metrics across ALL parametrizations to find the problematic one
"""
import pickle
import numpy as np
from pathlib import Path
from compare_parametrizations import ParametrizationComparator

print("Checking ALL metrics across ALL parametrizations...\n")

comparator = ParametrizationComparator(results_base_dir='results')

# Load ALL parametrizations
robustness_dirs = list(Path('results').glob('robustness_*'))
print(f"Found {len(robustness_dirs)} total parametrizations\n")

# Group by cost type
cost_groups = {}
for param_dir in sorted(robustness_dirs):  # ALL parametrizations this time
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

print("Parametrizations by cost type:")
for cost_type, params in cost_groups.items():
    print(f"  {cost_type}: {len(params)} parametrizations")

# Check each cost type
for cost_type, params in cost_groups.items():
    print(f"\n{'='*80}")
    print(f"CHECKING {cost_type.upper()} ({len(params)} parametrizations)")
    print(f"{'='*80}\n")

    temporal_coefficients = {}
    panel_coefficients = {}

    # Test ALL metrics
    for metric_name in ['gini_quantiles_avg', 'market_share_quantiles_avg', 'mean_members_avg', 'num_cong_avg']:
        print(f"\nMetric: {metric_name}")

        for param_info in params:
            param_name = param_info['name']
            data = param_info['data']

            # Prepare param_data
            shares = sorted(data.keys())
            param_data = {
                'shares': shares,
                'metrics': {}
            }

            # Extract metric
            param_data['metrics'][metric_name] = {}
            for share in shares:
                if metric_name in data[share]:
                    param_data['metrics'][metric_name][share] = data[share][metric_name]

            # Run sensitivity analysis
            results, _ = comparator.fit_alpha_sensitivity_models(param_data, metric_name)

            if results:
                # Check if results is dict with quantiles or dict with temporal/panel
                if any(q in results for q in ['10th', '25th', '50th', '75th', '90th', '99th', '100th']):
                    # Quantile-based metric
                    for quantile_name, quantile_data in results.items():
                        if isinstance(quantile_data, dict):
                            if 'temporal' in quantile_data:
                                temporal_data = quantile_data['temporal']
                                for coeff_name in ['linear_beta1']:
                                    if coeff_name in temporal_data:
                                        key = f"{metric_name}_{quantile_name}_temporal_{coeff_name}"
                                        if key not in temporal_coefficients:
                                            temporal_coefficients[key] = []
                                        temporal_coefficients[key].append(temporal_data[coeff_name])

                            if 'panel' in quantile_data:
                                panel_data = quantile_data['panel']
                                for coeff_name in ['linear_beta1']:
                                    if coeff_name in panel_data:
                                        key = f"{metric_name}_{quantile_name}_panel_{coeff_name}"
                                        if key not in panel_coefficients:
                                            panel_coefficients[key] = []
                                        panel_coefficients[key].append(panel_data[coeff_name])
                else:
                    # Scalar metric
                    if 'temporal' in results:
                        temporal_data = results['temporal']
                        for coeff_name in ['linear_beta1']:
                            if coeff_name in temporal_data:
                                key = f"{metric_name}_temporal_{coeff_name}"
                                if key not in temporal_coefficients:
                                    temporal_coefficients[key] = []
                                temporal_coefficients[key].append(temporal_data[coeff_name])

                    if 'panel' in results:
                        panel_data = results['panel']
                        for coeff_name in ['linear_beta1']:
                            if coeff_name in panel_data:
                                key = f"{metric_name}_panel_{coeff_name}"
                                if key not in panel_coefficients:
                                    panel_coefficients[key] = []
                                panel_coefficients[key].append(panel_data[coeff_name])

    # Check for problems
    print(f"\nChecking temporal coefficients:")
    problems_found = False
    for key, coefficient_arrays in temporal_coefficients.items():
        shapes = [np.array(arr).shape for arr in coefficient_arrays]
        unique_shapes = set(shapes)

        if len(unique_shapes) > 1:
            print(f"  ❌ {key}: {len(unique_shapes)} different shapes")
            print(f"     Shapes: {unique_shapes}")
            problems_found = True

    if not problems_found:
        print(f"  ✅ All temporal coefficients have consistent shapes")

    print(f"\nChecking panel coefficients:")
    problems_found = False
    for key, coefficient_scalars in panel_coefficients.items():
        are_scalars = all(np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0)
                          for val in coefficient_scalars)
        if not are_scalars:
            print(f"  ❌ {key}: contains non-scalars!")
            types = [type(val).__name__ for val in coefficient_scalars]
            shapes = [np.array(val).shape if hasattr(val, 'shape') else 'N/A' for val in coefficient_scalars]
            print(f"     Types: {set(types)}")
            print(f"     Shapes: {set(shapes)}")
            problems_found = True

    if not problems_found:
        print(f"  ✅ All panel coefficients are scalars")
