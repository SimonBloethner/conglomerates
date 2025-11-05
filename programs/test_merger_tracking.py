#!/usr/bin/env python3
"""
Test script to verify merger tracking functionality
"""

import numpy as np
import collaborative_growth
import matplotlib.pyplot as plt

def test_merger_tracking():
    """Test merger tracking with different share values"""
    print("=== TESTING MERGER TRACKING ===")
    
    # Test parameters (smaller for quick testing)
    markets = 5
    firms_per_market = 5
    steps = 20
    total_firms = markets * firms_per_market
    merge_thresh = 0.1  # Higher probability for testing
    comparison = 4
    break_thresh = 0.85
    lookback = 10
    ramp = 5
    proportional = False
    b0 = 0.00001
    b1 = 1.2
    
    # Test different share values
    shares_to_test = [0.0, 0.02, 0.1, 0.2]
    
    results = {}
    
    for share in shares_to_test:
        print(f"\nTesting share = {share}")
        
        params = [markets, firms_per_market, steps, share, total_firms, merge_thresh, 
                 comparison, break_thresh, proportional, lookback, b0, b1]
        
        np.random.seed(42)  # For reproducibility
        res = collaborative_growth.model(params=params)
        
        # Unpack results including new merger data
        mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, \
        max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, \
        avg_ranks, mergers_per_period = res
        
        # Analyze merger data
        total_mergers = np.sum(mergers_per_period)
        max_mergers_per_period = np.max(mergers_per_period)
        avg_mergers_per_period = np.mean(mergers_per_period)
        periods_with_mergers = np.sum(mergers_per_period > 0)
        
        results[share] = {
            'mergers_per_period': mergers_per_period,
            'total_mergers': total_mergers,
            'max_mergers_per_period': max_mergers_per_period,
            'avg_mergers_per_period': avg_mergers_per_period,
            'periods_with_mergers': periods_with_mergers,
            'merger_rate': total_mergers / (steps * firms_per_market * markets * merge_thresh) if share > 0 else 0
        }
        
        print(f"  Total mergers: {total_mergers}")
        print(f"  Max mergers per period: {max_mergers_per_period}")
        print(f"  Avg mergers per period: {avg_mergers_per_period:.2f}")
        print(f"  Periods with mergers: {periods_with_mergers}/{steps}")
        print(f"  Expected vs actual ratio: {results[share]['merger_rate']:.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Merger Frequency Analysis')
    
    # Plot 1: Mergers per period over time
    for i, (share, data) in enumerate(results.items()):
        color = plt.cm.viridis(i / len(results))
        axes[0, 0].plot(data['mergers_per_period'], label=f'α={share}', color=color, alpha=0.7)
    axes[0, 0].set_title('Mergers per Period Over Time')
    axes[0, 0].set_xlabel('Period')
    axes[0, 0].set_ylabel('Number of Mergers')
    axes[0, 0].legend()
    
    # Plot 2: Total mergers by share value
    shares = list(results.keys())
    total_mergers = [results[s]['total_mergers'] for s in shares]
    axes[0, 1].bar(range(len(shares)), total_mergers)
    axes[0, 1].set_title('Total Mergers by Share Value')
    axes[0, 1].set_xlabel('Share Value (α)')
    axes[0, 1].set_ylabel('Total Mergers')
    axes[0, 1].set_xticks(range(len(shares)))
    axes[0, 1].set_xticklabels([f'{s}' for s in shares])
    
    # Plot 3: Average mergers per period
    avg_mergers = [results[s]['avg_mergers_per_period'] for s in shares]
    axes[1, 0].bar(range(len(shares)), avg_mergers)
    axes[1, 0].set_title('Average Mergers per Period')
    axes[1, 0].set_xlabel('Share Value (α)')
    axes[1, 0].set_ylabel('Avg Mergers per Period')
    axes[1, 0].set_xticks(range(len(shares)))
    axes[1, 0].set_xticklabels([f'{s}' for s in shares])
    
    # Plot 4: Merger activity distribution
    for i, (share, data) in enumerate(results.items()):
        if data['total_mergers'] > 0:  # Only plot if there were mergers
            color = plt.cm.viridis(i / len(results))
            axes[1, 1].hist(data['mergers_per_period'], bins=10, alpha=0.6, 
                           label=f'α={share}', color=color, density=True)
    axes[1, 1].set_title('Distribution of Mergers per Period')
    axes[1, 1].set_xlabel('Mergers per Period')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('merger_tracking_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary analysis
    print("\n=== SUMMARY ===")
    print("Expected behavior:")
    print("- α=0: No mergers (sharing disabled)")
    print("- α>0: Mergers increase with sharing benefits")
    print("\nObserved:")
    for share in shares_to_test:
        print(f"α={share}: {results[share]['total_mergers']} total mergers")
    
    return results

if __name__ == "__main__":
    test_merger_tracking()