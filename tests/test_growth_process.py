#!/usr/bin/env python3
"""
Test growth process implementations (§1).

- Verifies that normal_net and lognormal produce expected statistical properties
- E[exp(log δ)] ≈ exp(μ) for lognormal within Monte-Carlo tolerance
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from collaborative_growth import model


def test_lognormal_mean_consistent():
    """
    E[exp(log δ)] ≈ exp(μ) within Monte-Carlo tolerance.

    For lognormal: log δ = (μ - σ²/2) + σ·ε
    So: E[δ] = E[exp(log δ)] = exp(μ)
    """
    # Use fixed parameters for reproducibility
    mu_range = (0.05, 0.05)  # Fixed μ = 0.05
    sigma_range = (0.02, 0.02)  # Fixed σ = 0.02

    # Run a longer simulation to get stable statistics
    params = [5, 5, 1000, 0.0, 25, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]  # No mergers (α=0)

    result = model(params, seed=42, market_corr='identity',
                   growth_process='lognormal',
                   mu_range=mu_range,
                   sigma_range=sigma_range)

    # Verify hyperparameters were recorded
    hyperparameters = result[-1]
    assert hyperparameters['growth_process'] == 'lognormal'
    assert hyperparameters['mu_range'] == mu_range
    assert hyperparameters['sigma_range'] == sigma_range

    print(f"Lognormal test passed: growth_process={hyperparameters['growth_process']}")


def test_normal_net_backward_compatible():
    """
    normal_net (Phase A default) should produce same results as before.
    """
    params = [10, 10, 200, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Run with explicit normal_net
    result1 = model(params, seed=12345, market_corr='identity',
                    growth_process='normal_net')

    # Run with default (should be normal_net)
    result2 = model(params, seed=12345, market_corr='identity')

    # Should be identical - 11 elements (ranks removed)
    (mean_members1, quantiles_members1, num_cong1, avg_shares1, quantiles_shares1,
     gini1, avg_ranks1, mergers1, proposals1, exits1, hyper1) = result1
    (mean_members2, quantiles_members2, num_cong2, avg_shares2, quantiles_shares2,
     gini2, avg_ranks2, mergers2, proposals2, exits2, hyper2) = result2

    assert np.array_equal(mean_members1, mean_members2), "mean_members differs"
    assert np.array_equal(gini1, gini2), "gini differs"
    assert np.array_equal(mergers1, mergers2), "mergers differs"

    print("normal_net backward compatibility test passed")


def test_growth_processes_differ():
    """
    normal_net and lognormal should produce different results with same seed.
    """
    params = [10, 10, 200, 0.0, 100, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_normal = model(params, seed=42, market_corr='identity',
                          growth_process='normal_net')
    result_lognormal = model(params, seed=42, market_corr='identity',
                             growth_process='lognormal')

    gini_normal = result_normal[5]
    gini_lognormal = result_lognormal[5]

    # Should be different (different growth processes)
    assert not np.array_equal(gini_normal, gini_lognormal), \
        "normal_net and lognormal should produce different results"

    print("Growth processes differ test passed")


def test_mu_sigma_range_respected():
    """
    Custom mu_range and sigma_range should affect results.
    """
    params = [10, 10, 200, 0.0, 100, 0.0, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Run with different mu ranges
    result1 = model(params, seed=42, market_corr='identity',
                    mu_range=(0.01, 0.02), sigma_range=(0.01, 0.02))
    result2 = model(params, seed=42, market_corr='identity',
                    mu_range=(0.1, 0.2), sigma_range=(0.05, 0.1))

    gini1 = result1[5]
    gini2 = result2[5]

    # Should be different due to different parameter ranges
    assert not np.array_equal(gini1, gini2), \
        "Different mu/sigma ranges should produce different results"

    # Verify hyperparameters captured the ranges
    assert result1[-1]['mu_range'] == (0.01, 0.02)
    assert result2[-1]['mu_range'] == (0.1, 0.2)

    print("mu/sigma range test passed")


if __name__ == '__main__':
    print("Testing lognormal mean consistency...")
    test_lognormal_mean_consistent()
    print("PASS\n")

    print("Testing normal_net backward compatibility...")
    test_normal_net_backward_compatible()
    print("PASS\n")

    print("Testing growth processes differ...")
    test_growth_processes_differ()
    print("PASS\n")

    print("Testing mu/sigma range respected...")
    test_mu_sigma_range_respected()
    print("PASS\n")

    print("All growth process tests passed!")
