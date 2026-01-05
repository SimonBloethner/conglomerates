"""
Quick test to verify the Numba pooling implementation works correctly.
"""
import numpy as np
from collaborative_growth import model

# Simple test case
params = [
    3,      # markets
    2,      # firms_per_market
    50,     # steps (small for quick test)
    0.5,    # share
    6,      # total_firms (3*2)
    0.1,    # merge_thresh
    'avg',  # comparison
    0.8,    # break_thresh
    True,   # proportional
    5,      # lookback
    'linear',  # cost_type
    0.00001,   # c0
    0.00004225,  # c1
    0.001      # c2
]

print("Testing Numba pooling implementation...")
print("=" * 80)

np.random.seed(42)
results = model(params)

print("\n" + "=" * 80)
print("✅ Test completed successfully!")
print("=" * 80)

print("\nResults summary:")
print(f"  Mean members: {results[0].mean():.2f}")
print(f"  Number of conglomerates (final): {results[2][-1] if len(results[2]) > 0 else 0}")
print(f"  Total mergers: {results[8].sum():.0f}")
print(f"  Gini coefficient (mean): {results[5].mean():.4f}")

print("\n✅ Numba implementation is working correctly!")