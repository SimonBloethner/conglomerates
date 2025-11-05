#!/usr/bin/env python3
"""
Debug script to compare α=0 model behavior with pure Brownian motion
"""

import numpy as np
import matplotlib.pyplot as plt
import collaborative_growth

def debug_alpha_zero_vs_brownian():
    """Compare α=0 model with pure Brownian motion"""
    
    print("=== DEBUGGING α=0 vs PURE BROWNIAN MOTION ===\n")
    
    # Parameters matching your counterfactual setup
    markets = 100
    firms_per_market = 100
    total_firms = markets * firms_per_market
    steps = 1000
    share = 0.0  # α=0 case
    merge_thresh = 0.05
    comparison = 4
    break_thresh = 0.85
    lookback = 50
    proportional = False
    b0 = 0.00001
    b1 = 1.2
    
    params = [markets, firms_per_market, steps, share, total_firms, merge_thresh, 
              comparison, break_thresh, proportional, lookback, b0, b1]
    
    print("1. RUNNING α=0 MODEL...")
    model_results = collaborative_growth.model(params)
    mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = model_results
    
    # Extract final market share metrics
    print(f"Market share array shape: {market_share.shape}")  # (markets, steps+1, firms_per_market)
    final_market_shares = market_share[:, -1, :].reshape(-1)  # All firms' final market shares
    
    max_market_share = np.max(final_market_shares)
    percentile_99 = np.percentile(final_market_shares, 99)
    percentile_90 = np.percentile(final_market_shares, 90)
    percentile_50 = np.percentile(final_market_shares, 50)
    percentile_10 = np.percentile(final_market_shares, 10)
    min_market_share = np.min(final_market_shares)
    
    print(f"α=0 MODEL MARKET SHARE RESULTS:")
    print(f"  Maximum market share: {max_market_share:.6f}")
    print(f"  99th percentile: {percentile_99:.6f}")
    print(f"  90th percentile: {percentile_90:.6f}")
    print(f"  50th percentile (median): {percentile_50:.6f}")
    print(f"  10th percentile: {percentile_10:.6f}")
    print(f"  Minimum market share: {min_market_share:.6f}")
    print(f"  Number of conglomerates (final): {num_cong[-1] if len(num_cong) > 0 else 0}")
    
    print(f"\nMARKET SHARE EVOLUTION OVER TIME:")
    time_steps = [0, 200, 400, 600, 800, 1000]
    print("Time | Max     | 99%     | 90%     | 50%     | 10%")
    print("-" * 55)
    for t in time_steps:
        shares_at_t = market_share[:, t, :].reshape(-1)
        max_t = np.max(shares_at_t)
        p99_t = np.percentile(shares_at_t, 99)
        p90_t = np.percentile(shares_at_t, 90)
        p50_t = np.percentile(shares_at_t, 50)
        p10_t = np.percentile(shares_at_t, 10)
        print(f"{t:4d} | {max_t:.5f} | {p99_t:.5f} | {p90_t:.5f} | {p50_t:.5f} | {p10_t:.5f}")
    
    print(f"\nEXPECTED vs ACTUAL:")
    print(f"Expected (your previous runs): Median→0, 90%→0, 99%→15%, Max→70%")
    print(f"Actual: Median={percentile_50:.5f}, 90%={percentile_90:.5f}, 99%={percentile_99:.5f}, Max={max_market_share:.5f}")
    
    if max_market_share < 0.1:
        print("🚨 PROBLEM: Max market share too low! Expected dominance ~70%")
    if percentile_50 > 0.001:
        print("🚨 PROBLEM: Median too high! Expected ~0%")
    
    print("\n2. ANALYZING RETURN STRUCTURE...")
    
    # Let's examine the return structure by running the model initialization
    np.random.seed(42)  # For reproducibility
    
    # Recreate the return generation from the model
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
    
    print(f"Growth variables (μ) range: {growth_vars[:, 0].min():.4f} to {growth_vars[:, 0].max():.4f}")
    print(f"Growth variables (σ) range: {growth_vars[:, 1].min():.4f} to {growth_vars[:, 1].max():.4f}")
    print(f"Mean μ: {growth_vars[:, 0].mean():.4f}")
    print(f"Mean σ: {growth_vars[:, 1].mean():.4f}")
    
    # Market correlation
    from scipy.stats import random_correlation
    eigen_vals = np.random.uniform(0.1, 3, markets)
    eigen_vals = eigen_vals * markets / eigen_vals.sum()
    market_corr = random_correlation.rvs(tuple(eigen_vals), random_state=np.random.default_rng())
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr
    
    print(f"Market correlation matrix diagonal: {np.diag(market_corr)[:5]}")  # Should be all 1s
    print(f"Market correlation off-diagonal sample: {market_corr[0, 1:6]}")
    print(f"Market correlation range: {market_corr[market_corr != 1].min():.4f} to {market_corr[market_corr != 1].max():.4f}")
    
    # Generate sample returns
    realizations = np.random.multivariate_normal(growth_vars[:, 0], market_cov, size=(10, firms_per_market)) + 1
    
    print(f"Sample returns range: {realizations.min():.4f} to {realizations.max():.4f}")
    print(f"Sample returns mean: {realizations.mean():.4f}")
    print(f"Sample returns std: {realizations.std():.4f}")
    
    print("\n3. COMPARING WITH PURE BROWNIAN MOTION...")
    
    # Pure Brownian motion simulation (matching our earlier test)
    np.random.seed(42)
    wealth = np.ones(total_firms)
    
    for step in range(steps):
        returns = np.random.normal(0.02, 0.05, total_firms)  # Independent returns
        wealth = wealth * (1 + returns)
        wealth = np.maximum(wealth, 0.001)
    
    # Normalize to market shares
    brownian_market_shares = wealth / wealth.sum()
    
    def gini_coefficient_calc(x):
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    brownian_gini = gini_coefficient_calc(brownian_market_shares)
    brownian_max = np.max(brownian_market_shares)
    brownian_99 = np.percentile(brownian_market_shares, 99)
    brownian_90 = np.percentile(brownian_market_shares, 90)
    brownian_50 = np.percentile(brownian_market_shares, 50)
    
    print(f"PURE BROWNIAN MOTION RESULTS:")
    print(f"  Final Gini coefficient: {brownian_gini:.4f}")
    print(f"  Maximum market share: {brownian_max:.4f}")
    print(f"  99th percentile: {brownian_99:.4f}")
    print(f"  90th percentile: {brownian_90:.4f}")
    print(f"  50th percentile: {brownian_50:.4f}")
    
    print("\n4. KEY DIFFERENCES:")
    print(f"  Gini difference: {abs(final_gini - brownian_gini):.4f}")
    print(f"  Max share difference: {abs(max_market_share - brownian_max):.4f}")
    print(f"  99th percentile difference: {abs(percentile_99 - brownian_99):.4f}")
    
    # Check if correlations are the issue
    print("\n5. TESTING CORRELATION HYPOTHESIS...")
    
    # Test with independent returns (no correlation)
    np.random.seed(42)
    wealth_independent = np.ones(total_firms)
    
    for step in range(steps):
        # Use model's μ and σ but make returns independent
        firm_returns = []
        for firm in range(total_firms):
            market_id = firm // firms_per_market
            mu = growth_vars[market_id, 0]
            sigma = growth_vars[market_id, 1]
            ret = np.random.normal(mu, sigma) + 1
            firm_returns.append(ret)
        
        wealth_independent = wealth_independent * np.array(firm_returns)
        wealth_independent = np.maximum(wealth_independent, 0.001)
    
    independent_market_shares = wealth_independent / wealth_independent.sum()
    independent_gini = gini_coefficient_calc(independent_market_shares)
    independent_max = np.max(independent_market_shares)
    
    print(f"INDEPENDENT RETURNS (same μ,σ as model):")
    print(f"  Final Gini: {independent_gini:.4f}")
    print(f"  Maximum market share: {independent_max:.4f}")
    
    print("\n6. CONCLUSION:")
    if abs(final_gini - independent_gini) < 0.1:
        print("❌ Correlation is NOT the main issue")
        print("🔍 Look for other factors (market structure, return distribution)")
    else:
        print("✅ Market correlations are dampening inequality!")
        print("💡 Solution: Make returns independent for α=0 case")

def pure_gbm_with_model_setup():
    """Pure GBM using exact same setup as model but no economics"""
    
    print("\n=== PURE GBM WITH MODEL SETUP ===\n")
    
    # Exact same parameters and setup as model
    markets = 100
    firms_per_market = 100
    steps = 1000
    total_firms = markets * firms_per_market
    
    # Exact same random parameter generation as model
    np.random.seed(42)
    
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
    
    from scipy.stats import random_correlation
    eigen_vals = np.random.uniform(0.1, 3, markets)
    eigen_vals = eigen_vals * markets / eigen_vals.sum()
    market_corr = random_correlation.rvs(tuple(eigen_vals), random_state=np.random.default_rng())
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr
    
    # Exact same realizations generation
    realizations = np.random.multivariate_normal(growth_vars[:, 0], market_cov, size=(steps, firms_per_market)) + 1
    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])
    
    print(f"Realizations shape: {realizations.shape}")
    print(f"Markets structure shape: {markets_structure.shape}")
    
    # Pure GBM: just multiply returns without any economics
    firm_wealth = np.ones((total_firms, steps + 1))  # Initialize all firms with wealth 1
    
    for step in range(steps):
        for firm in range(total_firms):
            # Use exact same indexing as the corrected model
            market_coords = markets_structure[firm, ::-1]  # [firm_in_market, market_id]
            firm_return = realizations[step, market_coords[0], market_coords[1]]
            
            # Pure multiplicative update (like solo firms in model)
            firm_wealth[firm, step + 1] = firm_wealth[firm, step] * firm_return
    
    # Calculate market shares exactly like the model
    market_share = np.zeros((markets, steps + 1, firms_per_market))
    
    for market in range(markets):
        # Get firms in this market
        firm_start = market * firms_per_market
        firm_end = (market + 1) * firms_per_market
        market_wealth = firm_wealth[firm_start:firm_end, :]  # Shape: (firms_per_market, steps+1)
        
        # Convert to market shares (normalize by total wealth in market at each time)
        market_totals = market_wealth.sum(axis=0)  # Sum across firms for each time step
        market_share[market, :, :] = (market_wealth / market_totals).T  # Transpose to get (steps+1, firms)
    
    # Extract final metrics
    final_market_shares = market_share[:, -1, :].reshape(-1)
    max_market_share = np.max(final_market_shares)
    percentile_99 = np.percentile(final_market_shares, 99)
    percentile_90 = np.percentile(final_market_shares, 90)
    percentile_50 = np.percentile(final_market_shares, 50)
    
    print(f"PURE GBM RESULTS:")
    print(f"  Maximum market share: {max_market_share:.6f}")
    print(f"  99th percentile: {percentile_99:.6f}")
    print(f"  90th percentile: {percentile_90:.6f}")
    print(f"  50th percentile: {percentile_50:.6f}")
    
    # Compare with α=0 model results
    print(f"\nCOMPARISON:")
    print(f"  Pure GBM Max: {max_market_share:.6f}")
    print(f"  α=0 Model Max: (run model to compare)")
    
    if max_market_share > 0.4:
        print("✅ Pure GBM shows expected extreme inequality")
    else:
        print("🚨 Even pure GBM shows dampened inequality - issue with setup")
    
    return market_share, firm_wealth

def test_aggregation_hypothesis():
    """Test if averaging explains the α=0 vs small α paradox"""
    
    print("\n=== TESTING AGGREGATION HYPOTHESIS ===\n")
    
    # Reduced parameters for faster testing
    markets = 20  # Reduced from 100
    firms_per_market = 50  # Reduced from 100
    steps = 300  # Reduced from 1000
    merge_thresh = 0.05
    comparison = 4
    break_thresh = 0.85
    lookback = 50
    proportional = False
    b0 = 0.00001
    b1 = 1.2
    total_firms = markets * firms_per_market
    
    alpha_values = [0.0, 0.02, 0.04]  # Test these three values
    n_experiments = 8  # More experiments for better statistics
    
    results = {}
    
    for alpha in alpha_values:
        print(f"\nTesting α = {alpha:.2f}")
        
        params = [markets, firms_per_market, steps, alpha, total_firms, merge_thresh, 
                 comparison, break_thresh, proportional, lookback, b0, b1]
        
        max_shares = []
        p99_values = []
        mean_ginis = []
        
        print("Exp | Max     | 99%     | Gini")
        print("-" * 35)
        
        for exp in range(n_experiments):
            # Use different seed for each experiment (like cluster does)
            np.random.seed(42 + exp * 1000 + int(alpha * 10000))
            
            model_results = collaborative_growth.model(params)
            mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares_raw, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = model_results
            
            # Extract metrics
            final_market_shares = market_share[:, -1, :].reshape(-1)
            max_share = np.max(final_market_shares)
            p99 = np.percentile(final_market_shares, 99)
            final_gini = np.mean(gini_coefficient[-1, :])
            
            max_shares.append(max_share)
            p99_values.append(p99)
            mean_ginis.append(final_gini)
            
            print(f"{exp:3d} | {max_share:.5f} | {p99:.5f} | {final_gini:.3f}")
        
        results[alpha] = {
            'max_shares': np.array(max_shares),
            'p99_values': np.array(p99_values),
            'mean_ginis': np.array(mean_ginis)
        }
        
        # Print statistics for this α
        print(f"Statistics for α={alpha:.2f}:")
        print(f"  Max shares: mean={np.mean(max_shares):.5f}, std={np.std(max_shares):.5f}")
        print(f"  99th %ile:  mean={np.mean(p99_values):.5f}, std={np.std(p99_values):.5f}")
        print(f"  Gini coeff: mean={np.mean(mean_ginis):.3f}, std={np.std(mean_ginis):.3f}")
    
    print("\n=== AGGREGATION ANALYSIS ===")
    
    # Compare averaged results (what cluster would report)
    print("\nCluster-style averaged results:")
    for alpha in alpha_values:
        max_data = results[alpha]['max_shares']
        p99_data = results[alpha]['p99_values']
        gini_data = results[alpha]['mean_ginis']
        
        avg_max = np.mean(max_data)
        avg_p99 = np.mean(p99_data)
        avg_gini = np.mean(gini_data)
        
        print(f"α={alpha:.2f}: Max={avg_max:.5f}, 99%={avg_p99:.5f}, Gini={avg_gini:.3f}")
    
    print("\n=== VARIANCE ANALYSIS ===")
    
    # Test the hypothesis: do small α cases have higher variance?
    alpha_0_var = np.var(results[0.0]['max_shares'])
    alpha_0_cv = np.std(results[0.0]['max_shares']) / np.mean(results[0.0]['max_shares'])
    
    print(f"α=0.0:")
    print(f"  Variance in max shares: {alpha_0_var:.6f}")
    print(f"  Coefficient of variation: {alpha_0_cv:.3f}")
    
    for alpha in [0.02, 0.04]:
        alpha_var = np.var(results[alpha]['max_shares'])
        alpha_cv = np.std(results[alpha]['max_shares']) / np.mean(results[alpha]['max_shares'])
        ratio = alpha_var / alpha_0_var
        cv_ratio = alpha_cv / alpha_0_cv
        
        print(f"α={alpha:.2f}:")
        print(f"  Variance: {alpha_var:.6f} (ratio to α=0: {ratio:.2f})")
        print(f"  CV: {alpha_cv:.3f} (ratio to α=0: {cv_ratio:.2f})")
        
        if ratio > 1.5:
            print(f"  ✅ Higher variance could explain averaging effect")
        else:
            print(f"  ❌ Variance not substantially higher")
    
    print("\n=== HYPOTHESIS TEST RESULTS ===")
    
    # Check if averaging explains the paradox
    alpha_0_avg = np.mean(results[0.0]['max_shares'])
    alpha_02_avg = np.mean(results[0.02]['max_shares'])
    alpha_04_avg = np.mean(results[0.04]['max_shares'])
    
    print(f"Averaged maximums (what cluster sees):")
    print(f"  α=0.0:  {alpha_0_avg:.5f}")
    print(f"  α=0.02: {alpha_02_avg:.5f}")
    print(f"  α=0.04: {alpha_04_avg:.5f}")
    
    if alpha_02_avg > alpha_0_avg and alpha_04_avg > alpha_0_avg:
        print("\n🚨 PARADOX CONFIRMED: Small α values show higher averaged maximums!")
        
        # Check if individual experiments tell different story
        alpha_0_max_individual = np.max(results[0.0]['max_shares'])
        alpha_02_max_individual = np.max(results[0.02]['max_shares'])
        
        print(f"\nIndividual experiment maximums:")
        print(f"  α=0.0 highest:  {alpha_0_max_individual:.5f}")
        print(f"  α=0.02 highest: {alpha_02_max_individual:.5f}")
        
        if alpha_0_max_individual > alpha_02_max_individual:
            print("✅ Individual α=0 experiments CAN be more extreme")
            print("💡 Averaging effect confirmed as explanation")
        else:
            print("❌ Even individual α=0 experiments are not more extreme")
    else:
        print("\n✅ No paradox observed in this test")
    
    return results

def test_theoretical_gbm_distribution():
    """Test against theoretical GBM wealth distribution predictions"""
    
    print("\n=== THEORETICAL GBM DISTRIBUTION TEST ===\n")
    
    # For geometric Brownian motion dW = μW dt + σW dB
    # After time T, log(W_T/W_0) ~ Normal(μT - σ²T/2, σ²T)
    # So W_T ~ LogNormal(μT - σ²T/2, σ²T)
    
    # Use model parameters
    T = 1000  # time steps
    μ = 0.055  # approximate mean from model
    σ = 0.03   # approximate volatility from model
    N = 10000  # number of firms
    
    print(f"Parameters: μ={μ:.3f}, σ={σ:.3f}, T={T}, N={N}")
    
    # Theoretical predictions for log-normal distribution
    log_mean = μ*T - (σ**2)*T/2
    log_std = σ * np.sqrt(T)
    
    print(f"Theoretical log-wealth: mean={log_mean:.2f}, std={log_std:.2f}")
    
    # Generate theoretical wealth distribution
    np.random.seed(42)
    log_wealth = np.random.normal(log_mean, log_std, N)
    theoretical_wealth = np.exp(log_wealth)
    theoretical_shares = theoretical_wealth / theoretical_wealth.sum()
    
    # Theoretical predictions for market shares
    def theoretical_gini_lognormal(μ, σ, T):
        """Theoretical Gini coefficient for log-normal distribution"""
        variance = σ**2 * T
        return 2 * scipy.stats.norm.cdf(np.sqrt(variance/2)) - 1
    
    try:
        import scipy.stats
        theoretical_gini = theoretical_gini_lognormal(μ, σ, T)
        print(f"Theoretical Gini coefficient: {theoretical_gini:.4f}")
    except ImportError:
        print("scipy not available for theoretical Gini calculation")
        theoretical_gini = None
    
    # Empirical results from theoretical distribution
    empirical_gini = gini_coefficient_calc(theoretical_shares)
    max_share = np.max(theoretical_shares)
    p99_share = np.percentile(theoretical_shares, 99)
    p90_share = np.percentile(theoretical_shares, 90)
    p50_share = np.percentile(theoretical_shares, 50)
    
    print(f"\nTheoretical GBM Results:")
    print(f"  Empirical Gini: {empirical_gini:.4f}")
    print(f"  Maximum share: {max_share:.6f}")
    print(f"  99th percentile: {p99_share:.6f}")
    print(f"  90th percentile: {p90_share:.6f}")
    print(f"  50th percentile: {p50_share:.6f}")
    
    # Compare with our model (reduced size for speed)
    print(f"\nRunning α=0 model for comparison...")
    
    markets = 50
    firms_per_market = 50
    steps = 200  # Reduced for speed
    total_firms = markets * firms_per_market
    
    params = [markets, firms_per_market, steps, 0.0, total_firms, 0.05, 4, 0.85, False, 50, 0.00001, 1.2]
    
    np.random.seed(42)
    model_results = collaborative_growth.model(params)
    mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares_raw, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = model_results
    
    final_market_shares = market_share[:, -1, :].reshape(-1)
    model_gini = gini_coefficient_calc(final_market_shares)
    model_max = np.max(final_market_shares)
    model_p99 = np.percentile(final_market_shares, 99)
    model_p90 = np.percentile(final_market_shares, 90)
    model_p50 = np.percentile(final_market_shares, 50)
    
    print(f"\nα=0 Model Results (T={steps}):")
    print(f"  Gini coefficient: {model_gini:.4f}")
    print(f"  Maximum share: {model_max:.6f}")
    print(f"  99th percentile: {model_p99:.6f}")
    print(f"  90th percentile: {model_p90:.6f}")
    print(f"  50th percentile: {model_p50:.6f}")
    
    print(f"\n=== COMPARISON ===")
    print(f"Gini - Theoretical: {empirical_gini:.4f}, Model: {model_gini:.4f}, Diff: {abs(empirical_gini-model_gini):.4f}")
    print(f"Max  - Theoretical: {max_share:.6f}, Model: {model_max:.6f}, Diff: {abs(max_share-model_max):.6f}")
    print(f"99%  - Theoretical: {p99_share:.6f}, Model: {model_p99:.6f}, Diff: {abs(p99_share-model_p99):.6f}")
    
    if abs(empirical_gini - model_gini) < 0.1:
        print("✅ Model matches theoretical GBM distribution well")
    else:
        print("❌ Model deviates significantly from theoretical GBM")
        if model_gini < empirical_gini:
            print("💡 Model inequality is LOWER than pure GBM - suggests dampening factors")
        else:
            print("💡 Model inequality is HIGHER than pure GBM - suggests amplifying factors")
    
    # Test extreme value theory predictions
    print(f"\n=== EXTREME VALUE ANALYSIS ===")
    
    # For log-normal, the maximum should scale as
    theoretical_max_expectation = np.sqrt(2 * log_std**2 * np.log(N))
    print(f"Theoretical max log-wealth (standardized): {theoretical_max_expectation:.2f}")
    
    # Convert to share expectation (very rough approximation)
    expected_max_wealth = np.exp(log_mean + theoretical_max_expectation * log_std)
    expected_total_wealth = N * np.exp(log_mean + log_std**2/2)  # E[W] for log-normal
    expected_max_share = expected_max_wealth / expected_total_wealth
    
    print(f"Very rough expected maximum share: {expected_max_share:.6f}")
    print(f"Actual theoretical maximum: {max_share:.6f}")
    print(f"Model maximum: {model_max:.6f}")
    
    return {
        'theoretical': theoretical_shares,
        'model': final_market_shares,
        'theoretical_gini': empirical_gini,
        'model_gini': model_gini
    }

def gini_coefficient_calc(x):
    """Calculate Gini coefficient"""
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

if __name__ == "__main__":
    debug_alpha_zero_vs_brownian()
    pure_gbm_with_model_setup()
    test_theoretical_gbm_distribution()
    test_aggregation_hypothesis()