"""Quick diagnostic to verify model behavior with referee fixes."""
import numpy as np
from collaborative_growth import model, seed_numba, get_cost_function_defaults

MARKETS = 30
FIRMS_PER_MARKET = 30
STEPS = 1000
total_firms = MARKETS * FIRMS_PER_MARKET

print(f"Quick diagnostic: {MARKETS}×{FIRMS_PER_MARKET}={total_firms} firms, {STEPS} steps\n")

print("| Cost Type | α | Mean Gini | Mergers | Proposals | Final Cong |")
print("|-----------|---|-----------|---------|-----------|------------|")

for cost_type in ['linear', 'quadratic', 'exponential', 'power_law']:
    defaults = get_cost_function_defaults(cost_type)
    
    for alpha in [0.0, 0.1, 0.3]:
        np.random.seed(42)
        seed_numba(42)
        
        params = [
            MARKETS, FIRMS_PER_MARKET, STEPS, alpha, total_firms,
            0.05, 4, 0.85, False, 50,
            cost_type, defaults['c0'], defaults['c1'], defaults['c2']
        ]
        
        result = model(params, seed=42, market_corr='identity')
        
        # Get metrics
        gini_per_market = result[5]  # Shape: (markets, steps)
        num_cong = result[2]         # List of arrays
        mergers = result[8]
        proposals = result[9]
        
        # Mean final Gini across markets
        mean_final_gini = np.mean(gini_per_market[:, -1])
        
        # Total mergers and proposals
        total_mergers = int(np.sum(mergers))
        total_proposals = int(np.sum(proposals))
        
        # Number of conglomerates (multi-firm entities) at end
        final_cong = len(num_cong[-1]) if hasattr(num_cong[-1], '__len__') else int(num_cong[-1])
        
        print(f"| {cost_type:11} | {alpha:.1f} | {mean_final_gini:9.4f} | {total_mergers:7d} | {total_proposals:9d} | {final_cong:10d} |")

print("\n## Key Validations:")
print("1. α=0 should have mergers=0 (since no benefit from sharing)")
print("2. Proposals should be >0 for all α (merger attempts happen)")
print("3. Gini should increase with α (more sharing → more concentration)")
print("\n✓ Model runs successfully with referee fixes")
