"""
Minimal test to verify that the realizations reshape optimization
produces identical indexing results without running the full simulation.

This bypasses the numpy.float128 issue by testing only the indexing logic.
"""
import numpy as np

def test_realizations_indexing():
    """
    Test that both indexing approaches extract the same values.
    """
    print("Testing realizations indexing logic...")
    print("="*80)

    # Small test case
    markets = 5
    firms_per_market = 4
    steps = 10
    total_firms = markets * firms_per_market

    # Create test data with known pattern
    # Shape: (steps, firms_per_market, markets)
    np.random.seed(42)
    mean_vector = np.random.rand(markets)
    cov_matrix = np.eye(markets) * 0.1

    # Generate realizations in original format
    realizations_original = np.random.multivariate_normal(
        mean_vector, cov_matrix, size=(steps, firms_per_market)
    ) + 1

    print(f"Original shape: {realizations_original.shape}")
    print(f"  (steps={steps}, firms_per_market={firms_per_market}, markets={markets})")

    # Create optimized version
    realizations_optimized = realizations_original.transpose(0, 2, 1).reshape(steps, total_firms)

    print(f"Optimized shape: {realizations_optimized.shape}")
    print(f"  (steps={steps}, total_firms={total_firms})")

    # Create markets_structure (same as in model)
    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])

    print(f"\nmarkets_structure shape: {markets_structure.shape}")
    print(f"First few entries: {markets_structure[:5]}")

    # Test 1: Individual firm access
    print("\n" + "-"*80)
    print("TEST 1: Individual firm access")
    print("-"*80)

    test_cases = [
        (0, 0),   # First firm in first market
        (2, 3),   # Firm 3 in market 2
        (4, 2),   # Firm 2 in last market
    ]

    all_match = True
    for market, firm_num in test_cases:
        firm_id = market * firms_per_market + firm_num

        # Original indexing (with reversed market_structure)
        original_value = realizations_original[5, firm_num, market]  # step=5

        # Optimized indexing
        optimized_value = realizations_optimized[5, firm_id]  # step=5

        match = np.allclose(original_value, optimized_value)
        all_match = all_match and match

        print(f"  Market {market}, Firm {firm_num} (firm_id={firm_id}):")
        print(f"    Original:  {original_value:.6f}")
        print(f"    Optimized: {optimized_value:.6f}")
        print(f"    Match: {match} {'✅' if match else '❌'}")

    # Test 2: Conglomerate access (multiple firms)
    print("\n" + "-"*80)
    print("TEST 2: Conglomerate access (multiple firms)")
    print("-"*80)

    # Simulate a conglomerate with firms from different markets
    conglomerate = [
        0,   # Market 0, Firm 0
        5,   # Market 1, Firm 1
        10,  # Market 2, Firm 2
        15   # Market 3, Firm 3
    ]

    step = 7
    indices = markets_structure[conglomerate, :]

    # Original indexing
    returns_original = realizations_original[step, indices[:, 1], indices[:, 0]]

    # Optimized indexing
    returns_optimized = realizations_optimized[step, conglomerate]

    match = np.allclose(returns_original, returns_optimized)
    all_match = all_match and match

    print(f"  Conglomerate firm IDs: {conglomerate}")
    print(f"  Indices (market, firm): {indices.tolist()}")
    print(f"  Original returns:  {returns_original}")
    print(f"  Optimized returns: {returns_optimized}")
    print(f"  Match: {match} {'✅' if match else '❌'}")
    print(f"  Max difference: {np.max(np.abs(returns_original - returns_optimized)):.2e}")

    # Test 3: Full array access pattern
    print("\n" + "-"*80)
    print("TEST 3: Complete array reconstruction")
    print("-"*80)

    # Verify we can reconstruct the entire original array from optimized
    reconstructed = np.zeros_like(realizations_original)

    for firm_id in range(total_firms):
        market = firm_id // firms_per_market
        firm_num = firm_id % firms_per_market
        reconstructed[:, firm_num, market] = realizations_optimized[:, firm_id]

    perfect_match = np.allclose(realizations_original, reconstructed)
    all_match = all_match and perfect_match

    max_diff = np.max(np.abs(realizations_original - reconstructed))
    print(f"  Perfect reconstruction: {perfect_match} {'✅' if perfect_match else '❌'}")
    print(f"  Maximum difference: {max_diff:.2e}")

    # Test 4: Edge cases
    print("\n" + "-"*80)
    print("TEST 4: Edge cases")
    print("-"*80)

    # First and last firms
    edge_cases = [
        (0, "First firm (0,0)"),
        (total_firms - 1, "Last firm ({},{})".format(markets-1, firms_per_market-1))
    ]

    for firm_id, description in edge_cases:
        market = firm_id // firms_per_market
        firm_num = firm_id % firms_per_market

        original_value = realizations_original[0, firm_num, market]
        optimized_value = realizations_optimized[0, firm_id]

        match = np.allclose(original_value, optimized_value)
        all_match = all_match and match

        print(f"  {description}:")
        print(f"    firm_id={firm_id}, market={market}, firm_num={firm_num}")
        print(f"    Match: {match} {'✅' if match else '❌'}")

    # Final summary
    print("\n" + "="*80)
    if all_match:
        print("✅ SUCCESS: All indexing tests passed!")
        print("The reshape optimization is mathematically correct.")
        print("Safe to use use_optimized_realizations=True in production.")
    else:
        print("❌ FAILURE: Some indexing tests failed!")
        print("DO NOT use the optimization until fixed.")
    print("="*80)

    return all_match

if __name__ == "__main__":
    success = test_realizations_indexing()
    exit(0 if success else 1)
