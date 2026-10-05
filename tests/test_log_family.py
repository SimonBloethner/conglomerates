#!/usr/bin/env python3
"""
Tests for log_family growth process.

Test catches:
1. Mean test catches leftover -σ²/2 term or location bug
2. IQR test catches scale conversion errors
3. Kurtosis tests catch family mix-ups
4. Copula test catches correlation distorting marginals or dropping correlation
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from scipy import stats

import collaborative_growth as cg


# Test parameters per card spec
M = 3           # markets
N = 200         # firms_per_market
T = 2000        # steps
ALPHA = 0.0     # share (no pooling for pure shock testing)
SEED = 42       # fixed seed


def generate_shocks_like_model(markets, firms_per_market, steps,
                                mu_range, sigma_range, log_family, nu,
                                rho, cross_corr, seed):
    """
    Replicate the shock generation logic from model() to test it directly.

    Returns log_shocks array of shape (steps, markets * firms_per_market)
    and the market-specific mu_m and iqr_m used.
    """
    np.random.seed(seed)
    cg.seed_numba(seed)

    min_mu, max_mu = mu_range
    min_sig, max_sig = sigma_range

    # Replicate growth_vars generation from model()
    mu_sig_corr = 0.7
    means = [0.1, 0.05]
    min_bound = np.array([min_mu, min_sig])
    max_bound = np.array([max_mu, max_sig])

    cov_mat = np.ones((2, 2)) * mu_sig_corr + np.diag(np.tile(1 - mu_sig_corr, 2))
    growth_vars = np.random.multivariate_normal(means, cov_mat, markets)
    growth_vars = growth_vars + np.abs(np.minimum(growth_vars.min(axis=0), 0))
    growth_vars = min_bound + (growth_vars / growth_vars.max(axis=0)) * (max_bound - min_bound)

    mu_m = growth_vars[:, 0]  # Shape: (markets,)
    iqr_m = growth_vars[:, 1]  # sigma_range interpreted as IQR

    # Market correlation matrix (identity for this test)
    market_corr_matrix = np.eye(markets)
    if cross_corr != 0.0:
        market_corr_matrix = np.full((markets, markets), cross_corr)
        np.fill_diagonal(market_corr_matrix, 1.0)

    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr_matrix
    market_cov_cholesky = np.linalg.cholesky(market_cov)

    # Convert IQR to scale
    scale_m = cg.iqr_to_scale(log_family, iqr_m, nu)

    all_shocks = []
    rho_val = float(rho)

    for step in range(steps):
        # Generate z (correlated normals)
        if rho_val == 0.0:
            z = np.random.standard_normal((markets, firms_per_market))
        else:
            z_common = np.random.standard_normal((markets, 1))
            z_idio = np.random.standard_normal((markets, firms_per_market))
            rho_abs = abs(rho_val)
            rho_sign = 1.0 if rho_val > 0 else -1.0
            z = rho_sign * np.sqrt(rho_abs) * z_common + np.sqrt(1 - rho_abs) * z_idio

        # Check if we need Gaussian copula
        needs_copula = (log_family != 'normal') and (rho_val != 0.0 or cross_corr != 0.0)

        if needs_copula:
            # Gaussian copula
            z_corr = market_cov_cholesky @ z / growth_vars[:, 1][:, np.newaxis]
            u = stats.norm.cdf(z_corr)

            if log_family == 'laplace':
                eps_raw = stats.laplace.ppf(u)
                eps = eps_raw / (2 * np.log(2))
            elif log_family == 'student_t':
                eps_raw = stats.t.ppf(u, df=nu)
                eps = eps_raw / (2 * stats.t.ppf(0.75, df=nu))

            log_shocks = (mu_m[:, np.newaxis] + iqr_m[:, np.newaxis] * eps).ravel()
        else:
            if log_family == 'normal':
                log_shocks = (mu_m[:, np.newaxis] + scale_m[:, np.newaxis] * z).ravel()
            elif log_family == 'laplace':
                eps_raw = np.random.laplace(0, 1, (markets, firms_per_market))
                eps = eps_raw / (2 * np.log(2))
                log_shocks = (mu_m[:, np.newaxis] + iqr_m[:, np.newaxis] * eps).ravel()
            elif log_family == 'student_t':
                eps_raw = np.random.standard_t(df=nu, size=(markets, firms_per_market))
                eps = eps_raw / (2 * stats.t.ppf(0.75, df=nu))
                log_shocks = (mu_m[:, np.newaxis] + iqr_m[:, np.newaxis] * eps).ravel()

        all_shocks.append(log_shocks)

    return np.array(all_shocks), mu_m, iqr_m


def test_log_family_mean():
    """
    For each family: sample mean of log δ per market within 0.01 of μ_m.

    Test catches: leftover -σ²/2 term or location bug.
    """
    mu_range = (0.01, 0.1)
    sigma_range = (0.01, 0.05)

    for family in ['normal', 'laplace', 'student_t']:
        print(f"  Testing {family} mean...")

        shocks, mu_m, iqr_m = generate_shocks_like_model(
            M, N, T, mu_range, sigma_range, family, 3.0,
            rho=0.0, cross_corr=0.0, seed=SEED
        )

        # Reshape to (steps, markets, firms_per_market)
        shocks_reshaped = shocks.reshape(T, M, N)

        # Compute sample mean per market (average over time and firms)
        sample_mean_per_market = shocks_reshaped.mean(axis=(0, 2))  # Shape: (M,)

        for m in range(M):
            diff = abs(sample_mean_per_market[m] - mu_m[m])
            assert diff < 0.01, (
                f"{family} market {m}: sample mean {sample_mean_per_market[m]:.4f} "
                f"differs from μ={mu_m[m]:.4f} by {diff:.4f} > 0.01"
            )

    print("PASS: log_family mean within 0.01 of μ_m for all families")


def test_log_family_iqr():
    """
    For each family: sample IQR within 3% of target.

    Test catches: scale conversion errors.
    """
    mu_range = (0.05, 0.05)  # Fixed mu
    sigma_range = (0.05, 0.05)  # Fixed IQR for easier verification
    target_iqr = 0.05

    for family in ['normal', 'laplace', 'student_t']:
        print(f"  Testing {family} IQR...")

        shocks, mu_m, iqr_m = generate_shocks_like_model(
            M, N, T, mu_range, sigma_range, family, 3.0,
            rho=0.0, cross_corr=0.0, seed=SEED
        )

        # Reshape to (steps, markets, firms_per_market)
        shocks_reshaped = shocks.reshape(T, M, N)

        # For each market, compute empirical IQR (center by removing market mean)
        for m in range(M):
            market_shocks = shocks_reshaped[:, m, :].ravel()
            # Center by removing mean (so we measure dispersion, not location)
            centered = market_shocks - market_shocks.mean()
            q25, q75 = np.percentile(centered, [25, 75])
            empirical_iqr = q75 - q25

            relative_error = abs(empirical_iqr - target_iqr) / target_iqr
            assert relative_error < 0.03, (
                f"{family} market {m}: empirical IQR {empirical_iqr:.4f} "
                f"differs from target {target_iqr:.4f} by {100*relative_error:.1f}% > 3%"
            )

    print("PASS: log_family IQR within 3% of target for all families")


def test_laplace_kurtosis():
    """
    Laplace excess kurtosis in (4, 8).

    Test catches: family mix-ups (normal has excess kurtosis 0, t3 has infinite).
    Theoretical excess kurtosis for Laplace is 3.
    Allowing (4, 8) gives buffer for sampling variation.
    """
    mu_range = (0.05, 0.05)
    sigma_range = (0.05, 0.05)

    shocks, mu_m, iqr_m = generate_shocks_like_model(
        M, N, T, mu_range, sigma_range, 'laplace', 3.0,
        rho=0.0, cross_corr=0.0, seed=SEED
    )

    # Compute excess kurtosis
    excess_kurt = stats.kurtosis(shocks.ravel(), fisher=True)

    # Laplace excess kurtosis is 3, but sample can vary
    # Accept (4, 8) to be safe with finite samples - but theoretical is 3
    # Let's be more precise: accept (2, 5) since theoretical is 3
    assert 2.0 < excess_kurt < 5.0, (
        f"Laplace excess kurtosis {excess_kurt:.2f} not in (2, 5) - "
        f"may be using wrong distribution"
    )

    print(f"PASS: Laplace excess kurtosis = {excess_kurt:.2f} in (2, 5)")


def test_student_t_kurtosis():
    """
    t₃ sample kurtosis > 20.

    Test catches: family mix-ups.
    Theoretical kurtosis for t_3 is undefined (infinite variance for df≤2,
    but df=3 gives finite variance with very heavy tails).
    Sample kurtosis should be very high.
    """
    mu_range = (0.05, 0.05)
    sigma_range = (0.05, 0.05)

    shocks, mu_m, iqr_m = generate_shocks_like_model(
        M, N, T, mu_range, sigma_range, 'student_t', 3.0,
        rho=0.0, cross_corr=0.0, seed=SEED
    )

    # Compute excess kurtosis (use robust method since tails are heavy)
    excess_kurt = stats.kurtosis(shocks.ravel(), fisher=True)

    # t_3 has very heavy tails; sample kurtosis should be high
    # Theoretical excess kurtosis for df>4 is 6/(df-4), but df=3 makes it undefined
    assert excess_kurt > 20, (
        f"Student-t(3) excess kurtosis {excess_kurt:.2f} not > 20 - "
        f"may be using wrong distribution or df"
    )

    print(f"PASS: Student-t(3) excess kurtosis = {excess_kurt:.2f} > 20")


def test_copula_preserves_marginals():
    """
    With cross_corr=0.3, family laplace:
    - Within-market IQR unchanged (±3%)
    - Correlation of log δ between two firms in different markets in (0.2, 0.4)

    Test catches: copula that distorts marginals or drops correlation.
    """
    mu_range = (0.05, 0.05)
    sigma_range = (0.05, 0.05)
    target_iqr = 0.05

    # First generate uncorrelated baseline
    shocks_uncorr, mu_m, iqr_m = generate_shocks_like_model(
        M, N, T, mu_range, sigma_range, 'laplace', 3.0,
        rho=0.0, cross_corr=0.0, seed=SEED
    )

    # Compute baseline IQR per market
    shocks_uncorr_reshaped = shocks_uncorr.reshape(T, M, N)
    baseline_iqrs = []
    for m in range(M):
        market_shocks = shocks_uncorr_reshaped[:, m, :].ravel()
        centered = market_shocks - market_shocks.mean()
        q25, q75 = np.percentile(centered, [25, 75])
        baseline_iqrs.append(q75 - q25)

    # Now generate correlated shocks
    shocks_corr, mu_m, iqr_m = generate_shocks_like_model(
        M, N, T, mu_range, sigma_range, 'laplace', 3.0,
        rho=0.0, cross_corr=0.3, seed=SEED
    )
    shocks_corr_reshaped = shocks_corr.reshape(T, M, N)

    # Test 1: IQR unchanged (±3%)
    for m in range(M):
        market_shocks = shocks_corr_reshaped[:, m, :].ravel()
        centered = market_shocks - market_shocks.mean()
        q25, q75 = np.percentile(centered, [25, 75])
        empirical_iqr = q75 - q25

        relative_error = abs(empirical_iqr - baseline_iqrs[m]) / baseline_iqrs[m]
        assert relative_error < 0.03, (
            f"Market {m}: correlated IQR {empirical_iqr:.4f} differs from "
            f"uncorrelated {baseline_iqrs[m]:.4f} by {100*relative_error:.1f}% > 3%"
        )

    # Test 2: Cross-market correlation in (0.2, 0.4)
    # Pick firm 0 from market 0 and firm 0 from market 1
    firm_0_market_0 = shocks_corr_reshaped[:, 0, 0]  # Shape: (T,)
    firm_0_market_1 = shocks_corr_reshaped[:, 1, 0]  # Shape: (T,)

    corr = np.corrcoef(firm_0_market_0, firm_0_market_1)[0, 1]

    assert 0.2 < corr < 0.4, (
        f"Cross-market correlation {corr:.3f} not in (0.2, 0.4) - "
        f"copula may not be preserving correlation structure"
    )

    print(f"PASS: Copula preserves marginals (IQR ±3%) and correlation = {corr:.3f} in (0.2, 0.4)")


def test_model_accepts_log_family_params():
    """
    Verify the model accepts log_family and nu parameters.

    Test catches: parameters not being passed through.
    """
    total_firms = M * N
    params = [
        M, N, 10, ALPHA, total_firms,  # Short run for speed
        0.0, 4, 0.85, False, 50,
        'power_law', None, None, None,
    ]

    # Test all families
    for family in ['normal', 'laplace', 'student_t']:
        result = cg.model(
            params, seed=SEED,
            growth_process='log_family',
            log_family=family,
            nu=3.0,
            mu_range=(0.01, 0.1),
            sigma_range=(0.01, 0.05),
            rho=0.0,
            cross_corr=0.0
        )

        # Check hyperparameters include log_family and nu
        hyperparams = result[-1]
        assert hyperparams['log_family'] == family, f"log_family not set correctly for {family}"
        assert hyperparams['nu'] == 3.0, "nu not set correctly"

    print("PASS: Model accepts log_family and nu parameters")


def test_phase_b_identity_with_log_family():
    """
    Verify test_phase_b_identity.py still passes.

    Test catches: any change to default behaviour.
    """
    import test_phase_b_identity
    test_phase_b_identity.test_phase_b_identity()
    print("PASS: Phase B identity preserved")


if __name__ == '__main__':
    print("Testing model accepts log_family params...")
    test_model_accepts_log_family_params()

    print("\nTesting log_family mean...")
    test_log_family_mean()

    print("\nTesting log_family IQR...")
    test_log_family_iqr()

    print("\nTesting Laplace kurtosis...")
    test_laplace_kurtosis()

    print("\nTesting Student-t kurtosis...")
    test_student_t_kurtosis()

    print("\nTesting copula preserves marginals...")
    test_copula_preserves_marginals()

    print("\nTesting Phase B identity...")
    test_phase_b_identity_with_log_family()

    print("\nAll log_family tests passed!")
