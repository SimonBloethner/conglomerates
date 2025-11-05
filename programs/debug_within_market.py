#!/usr/bin/env python3
"""
Debug within-market inequality patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import collaborative_growth

def debug_within_market_inequality():
    """Debug why within-market inequality is lower than expected"""
    
    print("=== DEBUGGING WITHIN-MARKET INEQUALITY ===\n")
    
    # Run α=0 model
    markets = 100
    firms_per_market = 100
    steps = 1000
    share = 0.0
    merge_thresh = 0.05
    comparison = 4
    break_thresh = 0.85
    lookback = 50
    proportional = False
    b0 = 0.00001
    b1 = 1.2
    total_firms = markets * firms_per_market
    
    params = [markets, firms_per_market, steps, share, total_firms, merge_thresh, 
              comparison, break_thresh, proportional, lookback, b0, b1]
    
    print("1. RUNNING α=0 MODEL AND EXAMINING INDIVIDUAL MARKETS...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    model_results = collaborative_growth.model(params)
    mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = model_results
    
    # Examine inequality within individual markets
    print(f"Market share array shape: {market_share.shape}")  # Should be (markets, steps+1, firms_per_market)
    
    # Look at final market shares in each market
    final_market_shares = market_share[:, -1, :]  # Shape: (markets, firms_per_market)
    
    print(f"Final market shares shape: {final_market_shares.shape}")
    
    # Calculate Gini for each market individually
    def gini_coefficient_calc(x):
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    market_ginis = []
    market_maxes = []
    for market_id in range(markets):
        market_shares = final_market_shares[market_id, :]
        market_gini = gini_coefficient_calc(market_shares)
        market_max = np.max(market_shares)
        market_ginis.append(market_gini)
        market_maxes.append(market_max)
    
    print(f"Individual market Ginis: mean={np.mean(market_ginis):.4f}, std={np.std(market_ginis):.4f}")
    print(f"Individual market max shares: mean={np.mean(market_maxes):.4f}, std={np.std(market_maxes):.4f}")
    print(f"Gini range: {np.min(market_ginis):.4f} to {np.max(market_ginis):.4f}")
    print(f"Max share range: {np.min(market_maxes):.4f} to {np.max(market_maxes):.4f}")
    
    # Compare with what we reported earlier
    overall_gini = gini_coefficient[-1, :].mean()
    overall_max = np.max(final_market_shares)
    print(f"Overall reported Gini: {overall_gini:.4f}")
    print(f"Overall max share: {overall_max:.4f}")
    
    print("\n2. EXAMINING A SINGLE MARKET IN DETAIL...")
    
    # Look at market 0 in detail
    market_0_shares = market_share[0, :, :]  # Time series for market 0
    print(f"Market 0 final shares: min={market_0_shares[-1, :].min():.6f}, max={market_0_shares[-1, :].max():.6f}")
    print(f"Market 0 final Gini: {gini_coefficient_calc(market_0_shares[-1, :]):.4f}")
    
    # Check wealth evolution in market 0
    # Need to extract raw wealth data
    print("\n3. EXAMINING RAW WEALTH EVOLUTION...")
    
    # Let's manually trace what happens to firms in market 0
    np.random.seed(42)  # Same seed as model
    
    # Recreate the model's random generation
    min_mu, max_mu = 0.01, 0.1
    min_sig, max_sig = 0.01, 0.05
    min_bound = np.array([min_mu, min_sig])
    max_bound = np.array([max_mu, max_sig])
    mu_sig_corr = 0.7
    means = [0.1, 0.05]
    
    cov_mat = np.ones((2, 2)) * mu_sig_corr + np.diag(np.tile(1 - mu_sig_corr, 2))
    growth_vars = np.random.multivariate_normal(means, cov_mat, markets)
    growth_vars = growth_vars + np.abs(np.minimum(growth_vars.min(axis=0), 0))
    growth_vars = min_bound + (growth_vars / growth_vars.max(axis=0)) * (max_bound - min_bound)
    
    # Market correlation and realizations
    from scipy.stats import random_correlation
    eigen_vals = np.random.uniform(0.1, 3, markets)
    eigen_vals = eigen_vals * markets / eigen_vals.sum()
    market_corr = random_correlation.rvs(tuple(eigen_vals), random_state=np.random.default_rng())
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr
    realizations = np.random.multivariate_normal(growth_vars[:, 0], market_cov, size=(steps, firms_per_market)) + 1
    
    print(f"Market 0 μ: {growth_vars[0, 0]:.4f}, σ: {growth_vars[0, 1]:.4f}")
    print(f"Market 0 returns: min={realizations[:, 0].min():.4f}, max={realizations[:, 0].max():.4f}, mean={realizations[:, 0].mean():.4f}")
    
    # Simulate wealth evolution for market 0 firms
    market_0_wealth = np.ones(firms_per_market)
    for step in range(steps):
        # Each firm in market 0 gets the same realization (within-market correlation = 1)
        market_0_wealth = market_0_wealth * realizations[step, :]
    
    # Convert to market shares
    market_0_final_shares = market_0_wealth / market_0_wealth.sum()
    manual_gini = gini_coefficient_calc(market_0_final_shares)
    manual_max = np.max(market_0_final_shares)
    
    print(f"Manual calculation - Market 0 Gini: {manual_gini:.4f}")
    print(f"Manual calculation - Market 0 max share: {manual_max:.4f}")
    
    print("\n4. COMPARING WITH TRUE INDEPENDENT BROWNIAN MOTION...")
    
    # True independent Brownian motion for 100 firms
    np.random.seed(42)
    independent_wealth = np.ones(firms_per_market)
    
    # Use same μ and σ as market 0, but independent draws
    mu_0 = growth_vars[0, 0]
    sigma_0 = growth_vars[0, 1]
    
    for step in range(steps):
        # Each firm gets independent return
        independent_returns = np.random.normal(mu_0, sigma_0, firms_per_market) + 1
        independent_wealth = independent_wealth * independent_returns
    
    independent_shares = independent_wealth / independent_wealth.sum()
    independent_gini = gini_coefficient_calc(independent_shares)
    independent_max = np.max(independent_shares)
    
    print(f"Independent Brownian - Gini: {independent_gini:.4f}")
    print(f"Independent Brownian - max share: {independent_max:.4f}")
    
    print("\n5. THE KEY INSIGHT...")
    
    if abs(manual_gini - independent_gini) < 0.05:
        print("✅ Within-market firms ARE getting independent returns!")
        print("🔍 The issue is elsewhere - possibly in aggregation across markets")
    else:
        print("❌ Within-market firms are NOT getting independent returns")
        print("💡 There's correlation within markets dampening inequality")
    
    print(f"\nReturn correlation within market 0:")
    market_0_returns = realizations[:, :10]  # First 10 firms in market 0, shape (steps, firms)
    print(f"Market 0 returns shape: {market_0_returns.shape}")
    
    # Check if all firms in market 0 get identical returns
    firm_0_returns = realizations[:, 0]  # Firm 0 in market 0
    firm_1_returns = realizations[:, 1]  # Firm 1 in market 0
    correlation = np.corrcoef(firm_0_returns, firm_1_returns)[0, 1]
    
    print(f"Correlation between firm 0 and firm 1 in market 0: {correlation:.4f}")
    
    # Check if returns are identical
    returns_identical = np.allclose(firm_0_returns, firm_1_returns)
    print(f"Returns identical for firms 0 and 1: {returns_identical}")
    
    if correlation > 0.99 or returns_identical:
        print("🚨 FOUND THE BUG: Firms within the same market have nearly identical returns!")
        print("💡 This prevents within-market inequality from developing")
    else:
        print("✅ Returns within markets are sufficiently independent")

if __name__ == "__main__":
    debug_within_market_inequality()