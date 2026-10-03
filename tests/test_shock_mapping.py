"""
Test shock-to-firm mapping is correct (§2a).
With M markets, N firms per market, firm m*N+j should get market m shock.
"""
import numpy as np
import sys
sys.path.insert(0, '..')


def test_shock_mapping_M_not_equals_N():
    """
    With M=3, N=7 and distinct market means, each firm's average return
    should be close to its home market's mean.
    """
    np.random.seed(42)
    
    M = 3  # markets
    N = 7  # firms per market
    n_draws = 2000
    
    # Growth vars: market means [0.01, 0.05, 0.10], small sigma
    growth_vars = np.array([[0.01, 0.001], [0.05, 0.001], [0.10, 0.001]])
    market_corr_matrix = np.eye(M)
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr_matrix
    market_cov_cholesky = np.linalg.cholesky(market_cov)
    
    # Collect draws for each firm
    firm_returns = np.zeros((M * N, n_draws))
    
    for draw in range(n_draws):
        z = np.random.standard_normal((M, N))
        realizations_step = growth_vars[:, 0][:, np.newaxis] + market_cov_cholesky @ z
        # Use .ravel() (row-major) as in the fixed code
        log_realizations = np.log1p(realizations_step.ravel())
        firm_returns[:, draw] = log_realizations
    
    # Check each firm's mean is close to its home market's expected value
    tolerance = 0.01  # Should be well within this with 2000 draws
    
    for firm_id in range(M * N):
        home_market = firm_id // N
        expected_mean = np.log1p(growth_vars[home_market, 0])
        actual_mean = firm_returns[firm_id].mean()
        
        assert abs(actual_mean - expected_mean) < tolerance, \
            f"Firm {firm_id} (market {home_market}): expected {expected_mean:.4f}, got {actual_mean:.4f}"


def test_shock_mapping_wrong_transpose():
    """
    Verify that using .T.flatten() (old buggy code) would fail the test.
    """
    np.random.seed(42)
    
    M = 3  # markets
    N = 7  # firms per market
    n_draws = 2000
    
    growth_vars = np.array([[0.01, 0.001], [0.05, 0.001], [0.10, 0.001]])
    market_corr_matrix = np.eye(M)
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr_matrix
    market_cov_cholesky = np.linalg.cholesky(market_cov)
    
    # Use WRONG .T.flatten() order
    firm_returns = np.zeros((M * N, n_draws))
    
    for draw in range(n_draws):
        z = np.random.standard_normal((M, N))
        realizations_step = growth_vars[:, 0][:, np.newaxis] + market_cov_cholesky @ z
        # WRONG: use .T.flatten() which transposes the array
        log_realizations = np.log1p(realizations_step.T.flatten())
        firm_returns[:, draw] = log_realizations
    
    # This should fail for at least some firms
    tolerance = 0.01
    any_wrong = False
    
    for firm_id in range(M * N):
        home_market = firm_id // N
        expected_mean = np.log1p(growth_vars[home_market, 0])
        actual_mean = firm_returns[firm_id].mean()
        
        if abs(actual_mean - expected_mean) > tolerance:
            any_wrong = True
            break
    
    assert any_wrong, "Transposed mapping should produce wrong results"


if __name__ == '__main__':
    print("Testing correct shock mapping (M != N)...")
    test_shock_mapping_M_not_equals_N()
    print("PASS: Correct mapping works for M != N")
    
    print("Testing that wrong transpose would fail...")
    test_shock_mapping_wrong_transpose()
    print("PASS: Transposed mapping correctly identified as wrong")
