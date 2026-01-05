#!/usr/bin/env python3
"""
Check polynomial_estimates and mobility metrics specifically
"""
import pickle
import numpy as np
from pathlib import Path
from compare_parametrizations import ParametrizationComparator

print("Checking polynomial_estimates and mobility metrics...\n")

comparator = ParametrizationComparator(results_base_dir='results')

# Load ALL parametrizations
robustness_dirs = list(Path('results').glob('robustness_*'))

# Group by cost type
cost_groups = {}
for param_dir in sorted(robustness_dirs):
    result_file = param_dir / 'counterfactual_results_final.pkl'

    if result_file.exists():
        with open(result_file, 'rb') as f:
            data = pickle.load(f)

        first_share = data[0.0]
        hyperparams = first_share.get('hyperparameters', {})
        cost_type = hyperparams.get('cost_type', 'unknown')

        if cost_type not in cost_groups:
            cost_groups[cost_type] = []

        cost_groups[cost_type].append({
            'name': param_dir.name,
            'data': data
        })

# Check polynomial_estimates for each cost type
for cost_type, params in cost_groups.items():
    print(f"\n{'='*80}")
    print(f"CHECKING {cost_type.upper()} - polynomial_estimates ({len(params)} parametrizations)")
    print(f"{'='*80}\n")

    poly_temporal = {}
    poly_panel = {}

    for param_info in params:
        param_name = param_info['name']
        data = param_info['data']

        # Prepare param_data
        shares = sorted(data.keys())
        param_data = {
            'shares': shares,
            'metrics': {}
        }

        # Extract poly_estimates_avg
        param_data['metrics']['poly_estimates_avg'] = {}
        for share in shares:
            if 'poly_estimates_avg' in data[share]:
                param_data['metrics']['poly_estimates_avg'][share] = data[share]['poly_estimates_avg']

        # Run polynomial sensitivity analysis
        poly_results = comparator.analyze_polynomial_sensitivity(param_data)

        if poly_results:
            print(f"  {param_name[:50]:<50} | Result keys: {list(poly_results.keys())}")

            # Check structure - it returns {'per_timestep': ..., 'panel': ...}
            if 'per_timestep' in poly_results:
                per_timestep = poly_results['per_timestep']
                for coeff_name, coeff_data in per_timestep.items():
                    # coeff_data is a dict with 'linear_beta1', etc. that are ARRAYS
                    for stat_name, stat_value in coeff_data.items():
                        if stat_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                            key = f"poly_{coeff_name}_per_timestep_{stat_name}"
                            if key not in poly_temporal:
                                poly_temporal[key] = []
                            poly_temporal[key].append(stat_value)

            if 'panel' in poly_results:
                panel = poly_results['panel']
                for coeff_name, coeff_data in panel.items():
                    # coeff_data is a dict with 'linear_beta1', etc. that are SCALARS
                    for stat_name, stat_value in coeff_data.items():
                        if stat_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                            key = f"poly_{coeff_name}_panel_{stat_name}"
                            if key not in poly_panel:
                                poly_panel[key] = []
                            poly_panel[key].append(stat_value)
        else:
            print(f"  {param_name[:50]:<50} | No poly results")

    # Check for problems
    print(f"\nChecking polynomial per_timestep (should be arrays):")
    problems_found = False
    for key, coefficient_arrays in poly_temporal.items():
        shapes = [np.array(arr).shape for arr in coefficient_arrays]
        unique_shapes = set(shapes)

        if len(unique_shapes) > 1:
            print(f"  ❌ {key}: {len(unique_shapes)} different shapes")
            print(f"     Shapes: {unique_shapes}")
            print(f"     Number of arrays: {len(coefficient_arrays)}")
            problems_found = True
        else:
            shape = list(unique_shapes)[0] if unique_shapes else None
            print(f"  ✅ {key}: all {len(coefficient_arrays)} arrays have shape {shape}")

    if not problems_found and poly_temporal:
        print(f"  ✅ All polynomial temporal coefficients have consistent shapes")
    elif not poly_temporal:
        print(f"  ⚠️  No polynomial temporal data found")

    print(f"\nChecking polynomial panel (should be scalars):")
    problems_found = False
    for key, coefficient_scalars in poly_panel.items():
        are_scalars = all(np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0)
                          for val in coefficient_scalars)
        if not are_scalars:
            print(f"  ❌ {key}: contains non-scalars!")
            types = [type(val).__name__ for val in coefficient_scalars]
            shapes = [np.array(val).shape if hasattr(val, 'shape') else 'N/A' for val in coefficient_scalars]
            print(f"     Types: {set(types)}")
            print(f"     Shapes: {set(shapes)}")
            problems_found = True
        else:
            print(f"  ✅ {key}: all {len(coefficient_scalars)} are scalars")

    if not problems_found and poly_panel:
        print(f"  ✅ All polynomial panel coefficients are scalars")
    elif not poly_panel:
        print(f"  ⚠️  No polynomial panel data found")
