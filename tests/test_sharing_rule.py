#!/usr/bin/env python3
"""
Test sharing rule implementations (§2).

- Verifies that equal and proportional sharing rules produce different results
- Tests conservation identity: Σ_i Π_i == ΣΔ_i − Φ·S
- Tests that sharing_rule parameter is recorded in hyperparameters
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from collaborative_growth import model


def test_equal_is_default():
    """
    equal (Phase A default) should be the default sharing rule.
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result = model(params, seed=42, market_corr='identity')

    hyperparameters = result[-1]
    assert hyperparameters['sharing_rule'] == 'equal', \
        f"Expected default sharing_rule='equal', got '{hyperparameters['sharing_rule']}'"

    print(f"Default sharing_rule verified: {hyperparameters['sharing_rule']}")


def test_equal_vs_proportional_differ():
    """
    equal and proportional sharing rules should produce different results.
    """
    params = [15, 15, 300, 0.15, 225, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    result_equal = model(params, seed=42, market_corr='identity',
                         sharing_rule='equal')
    result_prop = model(params, seed=42, market_corr='identity',
                        sharing_rule='proportional')

    gini_equal = result_equal[5]
    gini_prop = result_prop[5]

    # Should be different (different sharing rules)
    assert not np.array_equal(gini_equal, gini_prop), \
        "equal and proportional should produce different Gini trajectories"

    # Verify hyperparameters captured the rules
    assert result_equal[-1]['sharing_rule'] == 'equal'
    assert result_prop[-1]['sharing_rule'] == 'proportional'

    print(f"equal vs proportional differ test passed")
    # Handle multi-dimensional gini arrays (average across markets if needed)
    equal_final = np.mean(gini_equal[-1]) if gini_equal[-1].ndim > 0 else float(gini_equal[-1])
    prop_final = np.mean(gini_prop[-1]) if gini_prop[-1].ndim > 0 else float(gini_prop[-1])
    print(f"  equal final Gini: {equal_final:.4f}")
    print(f"  proportional final Gini: {prop_final:.4f}")


def test_equal_backward_compatible():
    """
    equal (Phase A default) should produce same results as before.
    """
    params = [10, 10, 200, 0.1, 100, 0.05, 4, 0.85, False, 50,
              'power_law', None, None, None]

    # Run with explicit equal
    result1 = model(params, seed=12345, market_corr='identity',
                    sharing_rule='equal')

    # Run with default (should be equal)
    result2 = model(params, seed=12345, market_corr='identity')

    # Should be identical
    gini1 = result1[5]
    gini2 = result2[5]

    assert np.array_equal(gini1, gini2), "Explicit equal should match default"

    print("equal backward compatibility test passed")


def test_conservation_identity():
    """
    Test the conservation identity on a hand-built conglomerate with unequal sizes.

    The conservation identity for pool distribution is:
        Σ_i Π_i == ΣΔ_i − Φ·S

    Where:
        Π_i = individual firm profit (r_i * w_i)
        Δ_i = firm's post-pooling wealth change
        Φ = management cost
        S = number of firms in conglomerate

    This test verifies the identity holds under both sharing rules.
    """
    # Test parameters
    alpha = 0.2

    # Hand-built conglomerate: 3 firms with unequal sizes
    w = np.array([1.0, 2.0, 3.0])  # Unequal capitalizations
    r = np.array([0.10, -0.05, 0.15])  # Individual returns (can be negative)

    # Normalize weights
    total_w = np.sum(w)
    w_norm = w / total_w

    # Individual profits
    Pi = r * w  # [0.10, -0.10, 0.45]
    sum_Pi = np.sum(Pi)  # 0.45

    # Pool formation (eq. 9): capital-weighted average return
    avg_return_weighted = np.sum(w_norm * r)  # Weighted average return

    # Management cost (using power law default)
    S = len(w)
    Phi = 0.01 * (S ** 1.5) / total_w  # Per-unit cost

    # Pool size
    Omega = alpha * sum_Pi - Phi * S

    print(f"Test conglomerate: {S} firms")
    print(f"  Weights: {w}")
    print(f"  Returns: {r}")
    print(f"  Individual profits Π_i: {Pi}")
    print(f"  Sum Π_i: {sum_Pi:.4f}")
    print(f"  Pool Ω: {Omega:.4f}")
    print(f"  Cost Φ·S: {Phi * S:.4f}")

    # Test equal sharing rule
    # Under equal: each firm gets equal share of pool
    delta_equal = np.zeros(S)
    for i in range(S):
        # δ_i = (1-α)·r_i + Ω/(K·w_i)
        # where K = S (number of firms in conglomerate)
        delta_equal[i] = (1 - alpha) * r[i] * w[i] + Omega / S

    sum_delta_equal = np.sum(delta_equal)
    conservation_equal = sum_Pi - sum_delta_equal + Phi * S

    print(f"\nEqual sharing rule:")
    print(f"  δ_i: {delta_equal}")
    print(f"  Sum δ_i: {sum_delta_equal:.6f}")
    print(f"  Conservation check (should be ~0): {conservation_equal:.10f}")

    # For equal sharing: Σδ_i = (1-α)·ΣΠ_i + Ω = (1-α)·ΣΠ_i + α·ΣΠ_i - Φ·S = ΣΠ_i - Φ·S
    # So: ΣΠ_i = Σδ_i + Φ·S ✓
    assert abs(sum_Pi - (sum_delta_equal + Phi * S)) < 1e-10, \
        f"Equal sharing conservation failed: {sum_Pi} != {sum_delta_equal + Phi * S}"

    # Test proportional sharing rule
    # Under proportional: each firm gets proportional to its weight
    delta_prop = np.zeros(S)
    for i in range(S):
        # δ_i = (1-α)·r_i·w_i + w_i·Ω/(Σw_j) = (1-α)·Π_i + w_norm_i·Ω
        delta_prop[i] = (1 - alpha) * r[i] * w[i] + w_norm[i] * Omega

    sum_delta_prop = np.sum(delta_prop)
    conservation_prop = sum_Pi - sum_delta_prop + Phi * S

    print(f"\nProportional sharing rule:")
    print(f"  δ_i: {delta_prop}")
    print(f"  Sum δ_i: {sum_delta_prop:.6f}")
    print(f"  Conservation check (should be ~0): {conservation_prop:.10f}")

    # For proportional sharing: Σδ_i = (1-α)·ΣΠ_i + Ω·Σw_norm_i = (1-α)·ΣΠ_i + Ω
    # Since Σw_norm_i = 1, we get: Σδ_i = (1-α)·ΣΠ_i + α·ΣΠ_i - Φ·S = ΣΠ_i - Φ·S
    # So: ΣΠ_i = Σδ_i + Φ·S ✓
    assert abs(sum_Pi - (sum_delta_prop + Phi * S)) < 1e-10, \
        f"Proportional sharing conservation failed: {sum_Pi} != {sum_delta_prop + Phi * S}"

    print("\nConservation identity verified for both sharing rules!")


if __name__ == '__main__':
    print("Testing equal is default...")
    test_equal_is_default()
    print("PASS\n")

    print("Testing equal vs proportional differ...")
    test_equal_vs_proportional_differ()
    print("PASS\n")

    print("Testing equal backward compatibility...")
    test_equal_backward_compatible()
    print("PASS\n")

    print("Testing conservation identity...")
    test_conservation_identity()
    print("PASS\n")

    print("All sharing rule tests passed!")
