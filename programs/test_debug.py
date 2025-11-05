import numpy as np
import collaborative_growth

# Test with a small parameter set
np.random.seed(42)

# Small test parameters
markets = 100
firms_per_market = 100
steps = 1000
share = 0.5
total_firms = markets * firms_per_market
merge_thresh = 0.05
comparison = 2
break_thresh = 0.85
lookback = 50
proportional = False
b0 = 0.000001
b1 = 1.05

model_params = [markets, firms_per_market, steps, share, total_firms, 
               merge_thresh, comparison, break_thresh, proportional, lookback, b0, b1]

print("Running model...")
results = collaborative_growth.model(model_params)

# Extract ranks
mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, \
max_shares, market_share, hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks = results

print(f"market_share shape: {market_share.shape}")
print(f"ranks shape: {ranks.shape}")
print(f"ranks data type: {type(ranks)}")
print(f"Expected dimensions: markets={markets}, firms_per_market={firms_per_market}, steps={steps}")

# Test accessing ranks
try:
    print("Testing rank access...")
    for step in range(min(3, ranks.shape[0])):  # Test first 3 steps
        for market in range(min(2, ranks.shape[1])):  # Test first 2 markets
            for firm in range(min(3, ranks.shape[2])):  # Test first 3 firms
                value = ranks[step, market, firm]
                print(f"ranks[{step}, {market}, {firm}] = {value}")
except Exception as e:
    print(f"Error accessing ranks: {str(e)}")
    print(f"ranks.shape = {ranks.shape}")
    print(f"ranks type = {type(ranks)}")
    if hasattr(ranks, 'dtype'):
        print(f"ranks dtype = {ranks.dtype}")