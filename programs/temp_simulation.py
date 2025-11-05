
import numpy as np
import collaborative_growth
import time

# Problematic parameter set 64
param_set = {
    "b0": 0.000001,
    "b1": 1.05, 
    "merge_thresh": 0.03,
    "markets": 100,
    "firms_per_market": 100,
    "share": 0.3
}

# Fixed parameters  
steps = 1000
comparison = 2
break_thresh = 0.85
lookback = 50
proportional = False
total_firms = param_set["markets"] * param_set["firms_per_market"]

model_params = [
    param_set["markets"],
    param_set["firms_per_market"], 
    steps,
    param_set["share"],
    total_firms,
    param_set["merge_thresh"],
    comparison,
    break_thresh,
    proportional,
    lookback,
    param_set["b0"],
    param_set["b1"]
]

# Run with specific seed
np.random.seed(6261)
start_time = time.time()
results = collaborative_growth.model(model_params)
end_time = time.time()

print(f"RESULT: Seed 6261 took {end_time - start_time:.2f} seconds")
