#!/usr/bin/env python3
"""
Test burn-in check functionality (§6).

- Verifies Geweke test on stationary vs non-stationary series
- Tests effective sample size computation
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from burnin_check import (
    geweke_test,
    effective_sample_size,
    analyze_burnin,
    recommend_burnin
)


def test_geweke_stationary():
    """
    Geweke test should pass for stationary series.
    """
    np.random.seed(42)
    # IID noise is stationary
    series = np.random.normal(0, 1, 500)

    z_score, p_value, converged = geweke_test(series)

    assert converged, f"Stationary series should converge, z={z_score:.2f}"
    assert p_value > 0.05, f"Stationary series should have high p-value, p={p_value:.4f}"

    print(f"Stationary series: z={z_score:.2f}, p={p_value:.4f}, converged={converged}")


def test_geweke_nonstationary():
    """
    Geweke test should fail for non-stationary series with drift.
    """
    np.random.seed(42)
    n = 500
    # Series with strong trend
    trend = np.linspace(0, 5, n)
    noise = np.random.normal(0, 0.1, n)
    series = trend + noise

    z_score, p_value, converged = geweke_test(series)

    assert not converged, f"Trending series should not converge, z={z_score:.2f}"

    print(f"Trending series: z={z_score:.2f}, p={p_value:.4f}, converged={converged}")


def test_effective_sample_size_iid():
    """
    ESS should equal actual sample size for IID series.
    """
    np.random.seed(42)
    series = np.random.normal(0, 1, 200)

    n_eff, autocorr_sum = effective_sample_size(series)

    # For IID, ESS should be close to actual n
    eff_ratio = n_eff / len(series)
    assert eff_ratio > 0.8, f"IID series should have high ESS ratio, got {eff_ratio:.2f}"

    print(f"IID series: n_eff={n_eff:.1f}, ratio={eff_ratio:.2f}")


def test_effective_sample_size_autocorr():
    """
    ESS should be less than actual sample size for autocorrelated series.
    """
    np.random.seed(42)
    n = 200

    # AR(1) process with high autocorrelation
    rho = 0.9
    series = np.zeros(n)
    series[0] = np.random.normal(0, 1)
    for t in range(1, n):
        series[t] = rho * series[t-1] + np.random.normal(0, np.sqrt(1 - rho**2))

    n_eff, autocorr_sum = effective_sample_size(series)

    # For high autocorrelation, ESS should be much less than n
    eff_ratio = n_eff / len(series)
    assert eff_ratio < 0.5, f"AR(1) series should have low ESS ratio, got {eff_ratio:.2f}"

    print(f"AR(1) series: n_eff={n_eff:.1f}, ratio={eff_ratio:.2f}")


def test_analyze_burnin_stationary():
    """
    Analysis should report convergence for stationary series.
    """
    np.random.seed(42)
    series = np.random.normal(0.5, 0.1, 300)

    results = analyze_burnin(series, 'test_metric', burnin_frac=0.1)

    assert results['geweke_converged'], "Stationary series should converge"
    assert results['mean_drift_pct'] < 10, f"Drift should be small, got {results['mean_drift_pct']:.2f}%"
    assert 0.8 < results['var_ratio'] < 1.25, f"Var ratio should be ~1, got {results['var_ratio']:.2f}"

    print(f"Burn-in analysis passed: drift={results['mean_drift_pct']:.2f}%, var_ratio={results['var_ratio']:.2f}")


def test_recommend_burnin():
    """
    Should recommend minimal burn-in for already stationary series.
    """
    np.random.seed(42)
    series = np.random.normal(1.0, 0.1, 300)

    recommended, analysis = recommend_burnin(series)

    # For stationary series, smallest burn-in should suffice
    assert recommended <= 30, f"Stationary series needs minimal burn-in, got {recommended}"
    assert len(analysis) > 0, "Should have analysis results"

    print(f"Recommended burn-in: {recommended} steps")


if __name__ == '__main__':
    print("Testing Geweke on stationary series...")
    test_geweke_stationary()
    print("PASS\n")

    print("Testing Geweke on non-stationary series...")
    test_geweke_nonstationary()
    print("PASS\n")

    print("Testing ESS for IID series...")
    test_effective_sample_size_iid()
    print("PASS\n")

    print("Testing ESS for autocorrelated series...")
    test_effective_sample_size_autocorr()
    print("PASS\n")

    print("Testing analyze_burnin on stationary series...")
    test_analyze_burnin_stationary()
    print("PASS\n")

    print("Testing burn-in recommendation...")
    test_recommend_burnin()
    print("PASS\n")

    print("All burn-in check tests passed!")
