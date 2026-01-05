"""
Validation test to verify that the realizations reshape optimization
produces identical results to the original implementation.
"""
import numpy as np
from collaborative_growth import model, get_cost_function_defaults

def run_comparison_test():
    """
    Run the same simulation with both old and new realizations access patterns.
    Compare outputs to verify they are identical.
    """
    print("Starting validation test for realizations reshape optimization...")
    print("="*80)

    # Use a small but meaningful test case
    hyperparameters = {
        'markets': 10,
        'firms_per_market': 10,
        'steps': 100,  # Shorter for quick testing
        'merge_thresh': 0.01,
        'comparison': 'geometric',
        'break_thresh': 1.0,
        'proportional': False,
        'lookback': 10,
        'cost_type': 'linear'
    }

    # Get cost function defaults
    defaults = get_cost_function_defaults(hyperparameters['cost_type'])

    # Test with alpha=0.1 (more interesting than alpha=0)
    alpha = 0.1
    total_firms = hyperparameters['markets'] * hyperparameters['firms_per_market']

    # Prepare parameters for the model function (14-parameter list format)
    params = [
        hyperparameters['markets'],
        hyperparameters['firms_per_market'],
        hyperparameters['steps'],
        alpha,
        total_firms,
        hyperparameters['merge_thresh'],
        hyperparameters['comparison'],
        hyperparameters['break_thresh'],
        hyperparameters['proportional'],
        hyperparameters['lookback'],
        hyperparameters['cost_type'],
        defaults['c0'],
        defaults['c1'],
        defaults['c2']
    ]

    # Set random seed for reproducibility
    np.random.seed(42)

    print(f"Running ORIGINAL version with alpha={alpha}, cost_type={hyperparameters['cost_type']}")
    print(f"Parameters: {hyperparameters['markets']} markets, {hyperparameters['firms_per_market']} firms, {hyperparameters['steps']} steps")
    print("-"*80)

    # Run original version (use_optimized_realizations=False)
    results_original = model(params, use_optimized_realizations=False)

    # Reset random seed to get identical random draws
    np.random.seed(42)

    print(f"\nRunning OPTIMIZED version with alpha={alpha}, cost_type={hyperparameters['cost_type']}")
    print("-"*80)

    # Run optimized version (use_optimized_realizations=True)
    results_optimized = model(params, use_optimized_realizations=True)

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    # Unpack results
    (mean_members_orig, quantiles_members_orig, num_cong_orig, avg_shares_orig,
     quantiles_shares_orig, market_share_orig, gini_orig, ranks_orig,
     avg_ranks_orig, mergers_orig, hyperparams_orig) = results_original

    (mean_members_opt, quantiles_members_opt, num_cong_opt, avg_shares_opt,
     quantiles_shares_opt, market_share_opt, gini_opt, ranks_opt,
     avg_ranks_opt, mergers_opt, hyperparams_opt) = results_optimized

    # Compare each output
    comparisons = [
        ("Mean members", mean_members_orig, mean_members_opt),
        ("Number of conglomerates", num_cong_orig, num_cong_opt),
        ("Average shares", avg_shares_orig, avg_shares_opt),
        ("Gini coefficient", gini_orig, gini_opt),
        ("Average ranks", avg_ranks_orig, avg_ranks_opt),
    ]

    all_identical = True

    for name, orig, opt in comparisons:
        if isinstance(orig, np.ndarray):
            identical = np.allclose(orig, opt, rtol=1e-10, atol=1e-12)
            max_diff = np.max(np.abs(orig - opt)) if orig.shape == opt.shape else float('inf')
            print(f"\n{name}:")
            print(f"  Shapes match: {orig.shape == opt.shape}")
            print(f"  Values identical (within tolerance): {identical}")
            if not identical:
                print(f"  Maximum difference: {max_diff}")
                all_identical = False
        else:
            identical = orig == opt
            print(f"\n{name}:")
            print(f"  Values identical: {identical}")
            if not identical:
                print(f"  Original: {orig}")
                print(f"  Optimized: {opt}")
                all_identical = False

    # Check array outputs
    array_comparisons = [
        ("Quantiles members", quantiles_members_orig, quantiles_members_opt),
        ("Quantiles shares", quantiles_shares_orig, quantiles_shares_opt),
        ("Market share", market_share_orig, market_share_opt),
        ("Ranks", ranks_orig, ranks_opt),
        ("Mergers per period", mergers_orig, mergers_opt),
    ]

    for name, orig, opt in array_comparisons:
        identical = np.allclose(orig, opt, rtol=1e-10, atol=1e-12)
        max_diff = np.max(np.abs(orig - opt)) if orig.shape == opt.shape else float('inf')
        print(f"\n{name}:")
        print(f"  Shape original: {orig.shape}")
        print(f"  Shape optimized: {opt.shape}")
        print(f"  Values identical (within tolerance): {identical}")
        if not identical:
            print(f"  Maximum difference: {max_diff}")
            all_identical = False

    print("\n" + "="*80)
    if all_identical:
        print("✅ SUCCESS: All outputs are identical!")
        print("The reshape optimization preserves the simulation logic perfectly.")
    else:
        print("❌ FAILURE: Outputs differ between original and optimized versions.")
        print("The optimization needs debugging.")
    print("="*80)

    return all_identical

if __name__ == "__main__":
    success = run_comparison_test()
    exit(0 if success else 1)
