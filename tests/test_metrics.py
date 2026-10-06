#!/usr/bin/env python3
"""
Unit tests for outcome metric functions.

Test catches:
1. HHI: Wrong formula or normalization
2. Aggregate HHI: Wrong control unit definition
3. effective_members: Wrong weight formula
4. Hill: Off-by-one in order statistic or log on wrong side
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from scipy import stats


def test_hhi_equal_shares():
    """
    HHI of N equal shares == 1/N.

    Test catches: wrong formula or normalization.
    """
    from collaborative_growth import compute_hhi

    for N in [2, 5, 10, 100]:
        shares = np.ones(N) / N
        hhi = compute_hhi(shares)
        expected = 1.0 / N
        assert abs(hhi - expected) < 1e-10, f"HHI of {N} equal shares = {hhi}, expected {expected}"

    print("PASS: HHI of N equal shares == 1/N")


def test_hhi_monopoly():
    """
    HHI of one firm owning all == 1.

    Test catches: wrong formula.
    """
    from collaborative_growth import compute_hhi

    # One firm with 100%
    shares = np.array([1.0])
    hhi = compute_hhi(shares)
    assert abs(hhi - 1.0) < 1e-10, f"HHI of monopoly = {hhi}, expected 1.0"

    # One firm with 100%, others with 0
    shares = np.array([1.0, 0.0, 0.0, 0.0])
    hhi = compute_hhi(shares)
    assert abs(hhi - 1.0) < 1e-10, f"HHI of monopoly with zeros = {hhi}, expected 1.0"

    print("PASS: HHI of one firm owning all == 1")


def test_aggregate_hhi_one_conglomerate():
    """
    Aggregate HHI with one conglomerate holding every firm == 1.

    Test catches: wrong control unit aggregation.
    """
    from collaborative_growth import compute_aggregate_hhi

    # 10 firms, all in one conglomerate (cong_id = 0)
    firm_sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    firm_conglom = np.zeros(10, dtype=np.int32)  # All in conglomerate 0

    hhi = compute_aggregate_hhi(firm_sizes, firm_conglom)
    assert abs(hhi - 1.0) < 1e-10, f"Aggregate HHI with one cong = {hhi}, expected 1.0"

    print("PASS: Aggregate HHI with one conglomerate == 1")


def test_aggregate_hhi_no_conglomerates():
    """
    Aggregate HHI with no conglomerates equals economy-wide firm HHI.

    Test catches: wrong handling of standalone firms.
    """
    from collaborative_growth import compute_aggregate_hhi, compute_hhi

    # 10 firms, all standalone (cong_id = -1)
    firm_sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    firm_conglom = np.full(10, -1, dtype=np.int32)  # All standalone

    # Aggregate HHI should equal firm-level HHI
    agg_hhi = compute_aggregate_hhi(firm_sizes, firm_conglom)
    firm_shares = firm_sizes / firm_sizes.sum()
    firm_hhi = compute_hhi(firm_shares)

    assert abs(agg_hhi - firm_hhi) < 1e-10, (
        f"Aggregate HHI = {agg_hhi}, firm HHI = {firm_hhi} - should be equal"
    )

    print("PASS: Aggregate HHI with no conglomerates equals firm HHI")


def test_effective_members_equal_sizes():
    """
    effective_members for K equal-size members == K.

    Test catches: wrong weight formula.
    """
    from collaborative_growth import compute_effective_members

    for K in [2, 5, 10, 20]:
        sizes = np.ones(K)
        eff = compute_effective_members(sizes)
        assert abs(eff - K) < 1e-10, f"Effective members for {K} equal = {eff}, expected {K}"

    print("PASS: effective_members for K equal-size members == K")


def test_effective_members_dominant():
    """
    effective_members for sizes (100,1,1,1,1) is < 1.1.

    Test catches: wrong concentration formula.
    """
    from collaborative_growth import compute_effective_members

    sizes = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    eff = compute_effective_members(sizes)

    # With one dominant firm at ~96% of total, effective members should be close to 1
    # w = [100/104, 1/104, 1/104, 1/104, 1/104] ≈ [0.962, 0.0096, ...]
    # sum(w²) ≈ 0.925, so 1/sum(w²) ≈ 1.08
    assert eff < 1.1, f"Effective members for (100,1,1,1,1) = {eff}, expected < 1.1"

    print(f"PASS: effective_members for (100,1,1,1,1) = {eff:.3f} < 1.1")


def test_hill_pareto():
    """
    Hill estimator on 10⁵ draws from Pareto(α=1.5) returns 1.5 ± 0.1.

    Test catches: off-by-one in order statistic or log on wrong side.

    Pareto with shape α has P(X > x) ~ x^{-α}, so Hill estimates α.
    """
    from collaborative_growth import hill_estimator

    np.random.seed(42)

    # Pareto with shape parameter α = 1.5
    alpha_true = 1.5
    n_samples = 100000

    # scipy.stats.pareto uses shape parameter b, where PDF ~ x^{-(b+1)}
    # For tail exponent α, we need b = α, giving P(X > x) ~ x^{-α}
    samples = stats.pareto.rvs(b=alpha_true, size=n_samples)

    # Hill estimator on top 10%
    alpha_est = hill_estimator(samples, k_fraction=0.1)

    error = abs(alpha_est - alpha_true)
    assert error < 0.1, f"Hill estimate = {alpha_est:.3f}, expected {alpha_true} ± 0.1"

    print(f"PASS: Hill on Pareto(1.5) = {alpha_est:.3f} (error = {error:.4f})")


def test_hill_2500_samples():
    """
    Hill estimator on 2,500 draws from Pareto(α=1.5) returns 1.5 ± 0.15.

    Test catches: implementation that fails with moderate sample sizes.
    """
    from collaborative_growth import hill_estimator

    np.random.seed(42)

    alpha_true = 1.5
    n_samples = 2500

    samples = stats.pareto.rvs(b=alpha_true, size=n_samples)
    alpha_est = hill_estimator(samples, k_fraction=0.1)

    error = abs(alpha_est - alpha_true)
    assert error < 0.15, f"Hill estimate = {alpha_est:.3f}, expected {alpha_true} ± 0.15"

    print(f"PASS: Hill on 2500 Pareto(1.5) = {alpha_est:.3f} (error = {error:.4f})")


def test_hill_5_draws_high_variance():
    """
    Hill estimator on 5 draws from Pareto(α=1.5), 50 repetitions,
    has standard deviation exceeding 0.4.

    Documents why pooling is needed: per-market Hill with N=50, k=5 is unstable.
    Test catches: a pooled version that silently still uses per-market data.
    """
    from collaborative_growth import hill_estimator

    alpha_true = 1.5
    n_reps = 50
    estimates = []

    for rep in range(n_reps):
        np.random.seed(rep)
        samples = stats.pareto.rvs(b=alpha_true, size=5)
        est = hill_estimator(samples, k_fraction=0.1)
        estimates.append(est)

    sd = np.std(estimates)
    print(f"5-observation SD across {n_reps} reps: {sd:.4f}")
    assert sd > 0.4, f"5-observation SD = {sd:.4f}, expected > 0.4"

    print(f"PASS: 5-observation Hill SD = {sd:.4f} > 0.4")


def test_pooled_hill_accuracy():
    """
    Pooled Hill on M=5 markets of N=400 shares all drawn from Pareto(α=1.5)
    returns 1.5 ± 0.1.

    Test catches: wrong pooling or wrong order statistic.
    """
    from collaborative_growth import hill_estimator

    np.random.seed(42)

    M = 5
    N = 400
    alpha_true = 1.5

    # Generate Pareto samples for each market
    all_sizes = stats.pareto.rvs(b=alpha_true, size=M * N)

    # Convert to market shares (fraction of total)
    market_shares = all_sizes / all_sizes.sum()

    # Pooled Hill on top 10% of all M×N = 2000 market shares
    alpha_est = hill_estimator(market_shares, k_fraction=0.1)

    error = abs(alpha_est - alpha_true)
    assert error < 0.1, f"Pooled Hill = {alpha_est:.3f}, expected {alpha_true} ± 0.1"

    print(f"PASS: Pooled Hill on M=5, N=400 Pareto(1.5) = {alpha_est:.3f} (error = {error:.4f})")


def test_pooled_hill_lower_variance():
    """
    Pooled Hill on M=50, N=50 Pareto(α=1.5) shares has std < 0.15 across 20 seeds;
    per-market version has std > 0.4.

    Demonstrates why the pooled estimator is used.
    Test catches: a "pooled" implementation that still averages per-market estimates.
    """
    from collaborative_growth import hill_estimator

    M = 50
    N = 50
    alpha_true = 1.5
    n_seeds = 20

    pooled_estimates = []
    permarket_stds = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        # Generate Pareto samples
        all_sizes = stats.pareto.rvs(b=alpha_true, size=M * N)

        # Pooled: Hill on all market shares
        market_shares = all_sizes / all_sizes.sum()
        pooled_hill = hill_estimator(market_shares, k_fraction=0.1)
        pooled_estimates.append(pooled_hill)

        # Per-market: Hill on each market's shares separately
        permarket_hills = []
        for m in range(M):
            market_sizes = all_sizes[m * N:(m + 1) * N]
            market_total = market_sizes.sum()
            market_share = market_sizes / market_total
            permarket_hills.append(hill_estimator(market_share, k_fraction=0.1))
        permarket_stds.append(np.std(permarket_hills))

    pooled_std = np.std(pooled_estimates)
    avg_permarket_std = np.mean(permarket_stds)

    print(f"Pooled std across {n_seeds} seeds: {pooled_std:.3f}")
    print(f"Avg per-market std within seed: {avg_permarket_std:.3f}")

    assert pooled_std < 0.15, f"Pooled Hill std = {pooled_std:.3f}, expected < 0.15"
    assert avg_permarket_std > 0.4, f"Per-market Hill std = {avg_permarket_std:.3f}, expected > 0.4"

    print(f"PASS: Pooled std = {pooled_std:.3f} < 0.15, per-market std = {avg_permarket_std:.3f} > 0.4")


def test_pooled_hill_same_distribution():
    """
    Pooled Hill on a constructed economy where every market has the same
    size vector equals the per-market Hill.

    Test catches: wrong share normalization in pooled implementation.
    """
    from collaborative_growth import hill_estimator

    # 5 markets, each with 10 firms, all having the same size vector
    M, N = 5, 10
    size_vector = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    all_sizes = np.tile(size_vector, M)  # Same distribution in each market

    # Pooled Hill (economy-wide shares)
    total_capital = all_sizes.sum()
    market_shares = all_sizes / total_capital
    pooled_hill = hill_estimator(market_shares, k_fraction=0.1)

    # Per-market Hill (should be identical since all markets have same distribution)
    market_total = size_vector.sum()
    per_market_shares = size_vector / market_total
    per_market_hill = hill_estimator(per_market_shares, k_fraction=0.1)

    # They should be equal when all markets have identical distributions
    assert abs(pooled_hill - per_market_hill) < 1e-10, (
        f"Pooled Hill = {pooled_hill:.6f}, per-market Hill = {per_market_hill:.6f} - "
        "should be equal when all markets have same distribution"
    )

    print(f"PASS: Pooled Hill = per-market Hill = {pooled_hill:.4f} (same distribution)")


def test_cong_capital_share_zero_alpha():
    """
    M=N=20, T=200, α=0 (no conglomerates ever): cong_capital_share is exactly 0.0
    at every recorded step and in the summary.

    Test catches: NaN returned when no conglomerate is active.
    """
    from collaborative_growth import model

    M, N, T = 20, 20, 200
    params = [
        M,                # markets
        N,                # firms_per_market
        T,                # steps
        0.0,              # share (alpha=0 - no conglomerates)
        M * N,            # total_firms
        0.05,             # merge_thresh
        4,                # comparison
        0.85,             # break_thresh
        False,            # proportional
        50,               # lookback
        'power_law',      # cost_type
        None, None, None, # c0, c1, c2
    ]

    result = model(params, seed=42, metric_every=10, burn_in=0)
    hyperparams = result[12]
    ccs = hyperparams['cong_capital_share']
    summary_ccs = hyperparams['summary']['cong_capital_share_median']

    # All values must be exactly 0.0 (not NaN)
    assert np.all(ccs == 0.0), (
        f"cong_capital_share has non-zero values: min={ccs.min()}, max={ccs.max()}"
    )
    assert not np.any(np.isnan(ccs)), "cong_capital_share contains NaN"
    assert summary_ccs == 0.0, f"Summary cong_capital_share_median = {summary_ccs}, expected 0.0"

    print(f"PASS: cong_capital_share = 0.0 at all {len(ccs)} steps with α=0")


def test_phase_b_identity_with_metrics():
    """
    test_phase_b_identity.py passes with new metrics.

    The identity test must compare only the keys the reference returns.
    """
    import test_phase_b_identity
    test_phase_b_identity.test_phase_b_identity()
    print("PASS: Phase B identity preserved with metrics")


if __name__ == '__main__':
    print("Testing HHI equal shares...")
    test_hhi_equal_shares()

    print("\nTesting HHI monopoly...")
    test_hhi_monopoly()

    print("\nTesting aggregate HHI one conglomerate...")
    test_aggregate_hhi_one_conglomerate()

    print("\nTesting aggregate HHI no conglomerates...")
    test_aggregate_hhi_no_conglomerates()

    print("\nTesting effective_members equal sizes...")
    test_effective_members_equal_sizes()

    print("\nTesting effective_members dominant...")
    test_effective_members_dominant()

    print("\nTesting Hill estimator on Pareto...")
    test_hill_pareto()

    print("\nTesting Hill on 2500 samples...")
    test_hill_2500_samples()

    print("\nTesting Hill 5-observation high variance...")
    test_hill_5_draws_high_variance()

    print("\nTesting pooled Hill accuracy...")
    test_pooled_hill_accuracy()

    print("\nTesting pooled Hill lower variance...")
    test_pooled_hill_lower_variance()

    print("\nTesting pooled Hill same distribution...")
    test_pooled_hill_same_distribution()

    print("\nTesting cong_capital_share with α=0...")
    test_cong_capital_share_zero_alpha()

    print("\nTesting Phase B identity...")
    test_phase_b_identity_with_metrics()

    print("\nAll metrics tests passed!")
