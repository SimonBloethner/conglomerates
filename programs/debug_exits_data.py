#!/usr/bin/env python3
"""
Debug script to diagnose why exits_per_period_avg shows "---" in tables
Tests three hypotheses:
1. Data doesn't exist in merged files
2. Data exists but is all zeros/NaN (no variation)
3. Data exists but controlled regression fails
"""
import pickle
import numpy as np
from pathlib import Path
import pandas as pd

def test_hypothesis_1_data_exists():
    """Check if exits_per_period_avg exists in the merged counterfactual files"""
    print("="*80)
    print("HYPOTHESIS 1: Does exits_per_period_avg exist in merged files?")
    print("="*80)

    results_dir = Path('results')

    # Look for robustness directories
    robustness_dirs = [d for d in results_dir.glob('robustness_*') if d.is_dir()]

    if not robustness_dirs:
        print(f"❌ No robustness directories found in {results_dir.absolute()}")
        print(f"\nSearching for ANY directories with counterfactual_results_final.pkl...")
        merged_files = list(results_dir.glob('*/counterfactual_results_final.pkl'))
        if merged_files:
            print(f"Found {len(merged_files)} files in non-robustness directories")
        else:
            print("❌ No merged files found anywhere!")
            return False
    else:
        merged_files = [d / 'counterfactual_results_final.pkl' for d in robustness_dirs if (d / 'counterfactual_results_final.pkl').exists()]

        if not merged_files:
            print(f"❌ No counterfactual_results_final.pkl files found in robustness directories!")
            print(f"   Found {len(robustness_dirs)} robustness directories but none have the merged file")
            return False

    print(f"Found {len(merged_files)} merged files\n")

    exists_count = 0
    missing_count = 0

    for merged_file in merged_files[:5]:  # Check first 5 files
        with open(merged_file, 'rb') as f:
            data = pickle.load(f)

        # Check first alpha value
        first_alpha = sorted(data.keys())[0]

        if 'exits_per_period_avg' in data[first_alpha]:
            exists_count += 1
            print(f"✅ {merged_file.parent.name}: exits_per_period_avg EXISTS")

            # Show some stats
            exits_data = data[first_alpha]['exits_per_period_avg']
            if isinstance(exits_data, np.ndarray):
                print(f"   Shape: {exits_data.shape}")
                print(f"   Min: {np.min(exits_data):.4f}, Max: {np.max(exits_data):.4f}")
                print(f"   Mean: {np.mean(exits_data):.4f}, Std: {np.std(exits_data):.4f}")
                print(f"   Non-zero values: {np.count_nonzero(exits_data)}/{len(exits_data)}")
        else:
            missing_count += 1
            print(f"❌ {merged_file.parent.name}: exits_per_period_avg MISSING")
            print(f"   Available keys: {sorted([k for k in data[first_alpha].keys() if 'exit' in k.lower() or 'merger' in k.lower() or 'period' in k.lower()])}")

    print(f"\nSummary: {exists_count} have data, {missing_count} missing")
    return exists_count > 0


def test_hypothesis_2_variation():
    """Check if exits_per_period_avg has sufficient variation across alpha values"""
    print("\n" + "="*80)
    print("HYPOTHESIS 2: Does exits_per_period_avg have variation across alpha?")
    print("="*80)

    results_dir = Path('results')

    # Look for robustness directories
    robustness_dirs = [d for d in results_dir.glob('robustness_*') if d.is_dir()]
    merged_files = [d / 'counterfactual_results_final.pkl' for d in robustness_dirs if (d / 'counterfactual_results_final.pkl').exists()]

    if not merged_files:
        # Fallback to any directory
        merged_files = list(results_dir.glob('*/counterfactual_results_final.pkl'))

    if not merged_files:
        print("❌ No merged files to check")
        return False

    # Load first file and check variation across alphas
    test_file = merged_files[0]
    print(f"Testing file: {test_file.parent.name}\n")

    with open(test_file, 'rb') as f:
        data = pickle.load(f)

    alphas = sorted(data.keys())
    print(f"Alpha values: {alphas}\n")

    # Collect exits data for each alpha
    exits_by_alpha = {}
    for alpha in alphas:
        if 'exits_per_period_avg' in data[alpha]:
            exits_by_alpha[alpha] = data[alpha]['exits_per_period_avg']

    if not exits_by_alpha:
        print("❌ No exits_per_period_avg data found")
        return False

    # Analyze variation
    print("Exits per period statistics by alpha:")
    print("-" * 80)

    all_means = []
    all_stds = []

    for alpha in sorted(exits_by_alpha.keys()):
        exits_data = exits_by_alpha[alpha]

        mean_val = np.mean(exits_data)
        std_val = np.std(exits_data)
        nonzero_frac = np.count_nonzero(exits_data) / len(exits_data)

        all_means.append(mean_val)
        all_stds.append(std_val)

        print(f"α = {alpha:.2f}:")
        print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"  Range: [{np.min(exits_data):.4f}, {np.max(exits_data):.4f}]")
        print(f"  Non-zero timesteps: {nonzero_frac*100:.1f}%")

    print("\n" + "-" * 80)
    print("Cross-alpha variation:")
    print(f"  Mean of means: {np.mean(all_means):.4f}")
    print(f"  Std of means: {np.std(all_means):.4f}")
    print(f"  Variation coefficient: {np.std(all_means)/np.mean(all_means) if np.mean(all_means) > 0 else 0:.4f}")

    # Check if variation is sufficient
    has_variation = np.std(all_means) > 0.001  # Small threshold

    if has_variation:
        print("✅ Data has variation across alpha values")
    else:
        print("❌ Data has NO variation across alpha values (might be all zeros or constant)")

    return has_variation


def test_hypothesis_3_regression():
    """Test if controlled regression can run on exits_per_period_avg data"""
    print("\n" + "="*80)
    print("HYPOTHESIS 3: Can controlled regression run on exits_per_period_avg?")
    print("="*80)

    # Simulate loading data like compare_parametrizations.py does
    from compare_parametrizations import ParametrizationComparator

    analyzer = ParametrizationComparator()

    # Discover parametrizations
    analyzer.discover_parametrizations()

    print(f"Found {len(analyzer.parametrizations)} parametrizations\n")

    # Load first parametrization
    first_param = list(analyzer.parametrizations.keys())[0]
    print(f"Testing with: {first_param}")

    try:
        param_data = analyzer.load_parametrization_data(first_param)

        # Check if exits_per_period_avg loaded
        if 'exits_per_period_avg' not in param_data['metrics']:
            print("❌ exits_per_period_avg not in loaded metrics")
            print(f"   Available metrics: {list(param_data['metrics'].keys())}")
            return False

        exits_data = param_data['metrics']['exits_per_period_avg']

        if not exits_data:
            print("❌ exits_per_period_avg is empty dict")
            return False

        print(f"✅ exits_per_period_avg loaded successfully")
        print(f"   Alpha values: {sorted(exits_data.keys())}")

        # Check data structure
        first_alpha = sorted(exits_data.keys())[0]
        sample_data = exits_data[first_alpha]

        print(f"   Data type: {type(sample_data)}")
        if isinstance(sample_data, np.ndarray):
            print(f"   Shape: {sample_data.shape}")
            print(f"   Has NaN: {np.any(np.isnan(sample_data))}")
            print(f"   All zero: {np.all(sample_data == 0)}")
            print(f"   Sample values (first 10): {sample_data[:10]}")

        # Try to run the simple panel analysis (without controls)
        print("\n   Attempting simple panel regression (no controls)...")

        shares = param_data['shares']
        n_timesteps = len(sample_data)

        # Collect panel data
        panel_data = []
        panel_alphas = []

        for alpha in shares:
            if alpha in exits_data and exits_data[alpha] is not None:
                data_array = exits_data[alpha]
                for t in range(len(data_array)):
                    val = data_array[t]
                    if not np.isnan(val):
                        panel_data.append(val)
                        panel_alphas.append(alpha)

        print(f"   Collected {len(panel_data)} observations")

        if len(panel_data) < 10:
            print(f"   ❌ Insufficient data points ({len(panel_data)} < 10)")
            return False

        # Try polynomial fit
        panel_data = np.array(panel_data)
        panel_alphas = np.array(panel_alphas)

        print(f"   Alpha range: [{np.min(panel_alphas):.2f}, {np.max(panel_alphas):.2f}]")
        print(f"   Exits range: [{np.min(panel_data):.4f}, {np.max(panel_data):.4f}]")

        # Linear fit
        linear_coeffs = np.polyfit(panel_alphas, panel_data, 1)
        print(f"   Linear fit: β₁={linear_coeffs[0]:.4f}, β₀={linear_coeffs[1]:.4f}")

        # Quadratic fit
        quad_coeffs = np.polyfit(panel_alphas, panel_data, 2)
        print(f"   Quadratic fit: β₂={quad_coeffs[0]:.4f}, β₁={quad_coeffs[1]:.4f}, β₀={quad_coeffs[2]:.4f}")

        print("   ✅ Simple regression works!")

    except Exception as e:
        print(f"❌ Error during regression test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_hypothesis_4_controlled_regression():
    """Test if controlled regression works (with hyperparameter controls)"""
    print("\n" + "="*80)
    print("HYPOTHESIS 4: Can controlled panel regression run on exits_per_period_avg?")
    print("="*80)

    from compare_parametrizations import ParametrizationComparator

    analyzer = ParametrizationComparator()

    print("Running controlled temporal regression analysis...")
    print("This will show if exits_per_period_avg appears in results\n")

    try:
        # First need to load all parametrizations
        print("Loading all parametrizations...")
        analyzer.discover_parametrizations()
        all_data = analyzer.load_all_parametrizations()

        if not all_data:
            print("❌ No parametrization data loaded")
            return False

        print(f"Loaded {len(all_data)} parametrizations\n")

        # Run the full controlled analysis
        results = analyzer.run_controlled_temporal_analysis(all_data, control_hyperparams=True)

        if results is None:
            print("❌ Controlled analysis returned None")
            return False

        # Check if exits_per_period_avg appears in any cost function results
        by_cost = results.get('by_cost_function', {})

        found_exits = False
        for cost_type, cost_data in by_cost.items():
            print(f"\nCost function: {cost_type}")

            # Check what metrics are available
            metrics_with_overall = []
            for metric_name in cost_data.keys():
                if isinstance(cost_data[metric_name], dict) and 'overall' in cost_data[metric_name]:
                    metrics_with_overall.append(metric_name)

            if 'exits_per_period_avg' in metrics_with_overall:
                found_exits = True
                print(f"  ✅ exits_per_period_avg found!")

                exits_results = cost_data['exits_per_period_avg']['overall']

                if 'quadratic' in exits_results:
                    quad = exits_results['quadratic']
                    print(f"     Quadratic results:")
                    print(f"       β₁ (alpha): {quad.get('alpha_coeff', 'N/A')}")
                    print(f"       β₂ (alpha²): {quad.get('alpha_sq_coeff', 'N/A')}")
                    print(f"       R²: {quad.get('r2', 'N/A')}")
                    print(f"       N obs: {quad.get('n_obs', 'N/A')}")
            else:
                print(f"  ❌ exits_per_period_avg NOT in results")
                print(f"     Available scalar metrics: {[m for m in metrics_with_overall if 'exit' in m.lower() or 'merger' in m.lower() or 'member' in m.lower() or 'num_cong' in m.lower()]}")

        if not found_exits:
            print("\n❌ exits_per_period_avg not found in ANY cost function results")
            print("   This means the controlled regression is failing or skipping this metric")
        else:
            print("\n✅ exits_per_period_avg found in controlled regression results")

        return found_exits

    except Exception as e:
        print(f"❌ Error during controlled regression: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DEBUGGING EXITS_PER_PERIOD_AVG MISSING DATA ISSUE")
    print("="*80 + "\n")

    # Test all hypotheses
    h1 = test_hypothesis_1_data_exists()

    if h1:
        h2 = test_hypothesis_2_variation()
        h3 = test_hypothesis_3_regression()
        h4 = test_hypothesis_4_controlled_regression()

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"H1 - Data exists in files: {'✅ PASS' if h1 else '❌ FAIL'}")
        print(f"H2 - Data has variation: {'✅ PASS' if h2 else '❌ FAIL'}")
        print(f"H3 - Simple regression works: {'✅ PASS' if h3 else '❌ FAIL'}")
        print(f"H4 - Controlled regression works: {'✅ PASS' if h4 else '❌ FAIL'}")

        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)

        if not h1:
            print("❌ ROOT CAUSE: Data doesn't exist in merged files")
            print("   ACTION: Check parallel_counterfactuals.py and merge_counterfactual_results.py")
        elif not h2:
            print("⚠️  LIKELY CAUSE: Data exists but has no variation (all zeros or constant)")
            print("   ACTION: Check if firms are actually exiting conglomerates in simulations")
        elif not h3:
            print("❌ ISSUE: Simple regression fails")
            print("   ACTION: Debug data structure or regression implementation")
        elif not h4:
            print("❌ ISSUE: Controlled regression fails")
            print("   ACTION: Debug the run_controlled_temporal_analysis() method")
        else:
            print("✅ All tests pass - exits_per_period_avg should work!")
            print("   If still showing '---', check table generation code")
    else:
        print("\n❌ Cannot proceed - data doesn't exist in merged files")